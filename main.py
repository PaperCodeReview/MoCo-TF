import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np
from common import set_seed
from common import get_logger
from common import get_session
from common import search_same
from common import create_stamp
from dataloader import set_dataset
from dataloader import DataLoader
from model import create_model
from model import momentum_update_model
from model import enqueue
from callback import OptionalLearningRateSchedule
from callback import create_callbacks

import tensorflow as tf


def main(args=None):
    set_seed()
    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
        args.stamp = create_stamp()

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))


    ##########################
    # Strategy
    ##########################
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    num_workers = strategy.num_replicas_in_sync
    assert args.batch_size % num_workers == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, num_workers))
    logger.info("GLOBAL BATCH SIZE : {}".format(args.batch_size))
    
    
    ##########################
    # Dataset
    ##########################
    trainset = set_dataset(args.data_path)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))


    ##########################
    # Model & Metric & Generator
    ##########################
    # metrics
    metrics = {
        'loss'          : tf.keras.metrics.Mean('loss', dtype=tf.float32),
        'acc1'          : tf.keras.metrics.TopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
        'acc5'          : tf.keras.metrics.TopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32),
        'total_loss'    : tf.keras.metrics.Mean('total_loss', dtype=tf.float32),
        'total_acc1'    : tf.keras.metrics.TopKCategoricalAccuracy(1, 'total_acc1', dtype=tf.float32),
        'total_acc5'    : tf.keras.metrics.TopKCategoricalAccuracy(5, 'total_acc5', dtype=tf.float32),    }
    with strategy.scope():
        encoder_q, encoder_k, queue = create_model(
            logger,
            backbone=args.backbone,
            img_size=args.img_size,
            dim=args.dim,
            K=args.num_negative, 
            mlp=args.mlp,
            snapshot=args.snapshot)

        if args.summary:
            encoder_q.summary()
            return

        # optimizer & loss
        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.0001)
        criterion = tf.keras.losses.categorical_crossentropy

        # generator
        train_generator = DataLoader(args, 'train', trainset, args.batch_size, num_workers).dataloader
        # for t in train_generator:
        #     print(t['query'].shape, tf.reduce_min(t['query']), tf.reduce_max(t['query']), t['key'].shape)
        
        train_generator = strategy.experimental_distribute_dataset(train_generator)

    csvlogger, train_writer = create_callbacks(args, metrics)
    logger.info("Build Model & Metrics")

    ##########################
    # READY Train
    ##########################
    train_iterator = iter(train_generator)
    
    def do_step(iterator, mode, queue):
        def get_loss(img_q, key, labels, N, C, K):
            query = encoder_q(img_q, training=True)
            query = tf.math.l2_normalize(query, axis=1)
            l_pos = tf.reshape(
                tf.matmul(
                    tf.reshape(query, [N, 1, C]), 
                    tf.reshape(key, [N, C, 1])), 
                [N, 1])                                 # l_pos : (256, 1)
            l_neg = tf.matmul(query, queue)             # l_neg : (256, 65536)
            logits = tf.concat((l_pos, l_neg), axis=1)  # logits : (256, 65536+1)
            logits /= args.temperature
            loss = criterion(labels, logits, from_logits=True)
            loss_mean = tf.nn.compute_average_loss(loss, global_batch_size=args.batch_size)
            return logits, loss, loss_mean

        def step_fn(from_iter):
            if args.shuffle_bn and num_workers > 1:
                x, unshuffle_idx = from_iter
                img_q, img_k = x['query'], x['key']
            else:
                img_q, img_k = from_iter['query'], from_iter['key']

            N = tf.shape(img_q)[0]
            C = tf.shape(queue)[0]
            K = tf.shape(queue)[1]
            labels = tf.one_hot(tf.zeros(N, dtype=tf.int32), args.num_negative+1) # labels : (256, 65536+1)

            key = encoder_k(img_k, training=False)
            key = tf.math.l2_normalize(key, axis=1)
            if args.shuffle_bn and num_workers > 1:
                def concat_fn(strategy, key_per_replica):
                    return tf.concat(key_per_replica.values, axis=0)
                replica_context = tf.distribute.get_replica_context()
                key_all_replica = replica_context.merge_call(concat_fn, args=(key,))
                unshuffle_idx_all_replica = replica_context.merge_call(concat_fn, args=(unshuffle_idx,))
                new_key_list = []
                for idx in unshuffle_idx_all_replica:
                    new_key_list.append(tf.expand_dims(key_all_replica[idx], axis=0))
                key_orig = tf.concat(tuple(new_key_list), axis=0)
                key = key_orig[(args.batch_size//num_workers)*(replica_context.replica_id_in_sync_group):
                               (args.batch_size//num_workers)*(replica_context.replica_id_in_sync_group+1)]

            if mode == 'train':
                with tf.GradientTape() as tape:
                    logits, loss, loss_mean = get_loss(img_q, key, labels, N, C, K)
                
                grads = tape.gradient(loss_mean, encoder_q.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, encoder_q.trainable_variables)))

            else:
                logits, loss, loss_mean = get_loss(img_q, key, labels, N, C, K)

            metrics['acc1'].update_state(labels, logits)
            metrics['acc5'].update_state(labels, logits)
            metrics['total_acc1'].update_state(labels, logits)
            metrics['total_acc5'].update_state(labels, logits)
            return key, loss

        new_key, loss_per_replica = strategy.run(step_fn, args=(next(iterator),))
        if num_workers == 1:
            loss_mean = tf.reduce_mean(loss_per_replica)
            return loss_mean, new_key
        else:
            loss_mean = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_replica, axis=0)
            return loss_mean, tf.concat(new_key.values, axis=0)
        

    ##########################
    # Train
    ##########################
    for epoch in range(initial_epoch, args.epochs):
        print('\nEpoch {}/{}'.format(epoch+1, args.epochs))
        print('Learning Rate : {}'.format(optimizer.learning_rate(optimizer.iterations)))

        # train
        print('Train')
        encoder_q.trainable = True
        progBar_train = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=metrics.keys())
        for step in range(steps_per_epoch):
            loss_mean, key = do_step(train_iterator, 'train', queue)
            metrics['loss'].update_state(loss_mean)
            metrics['total_loss'].update_state(loss_mean)
            queue = enqueue(queue, key, K=args.num_negative)
            encoder_k = momentum_update_model(encoder_q, encoder_k, m=args.momentum)
            progBar_train.update(step, values=[(k, v.result()) for k, v in metrics.items()])
            
            if args.tensorboard and args.tb_interval > 0:
                if (epoch*steps_per_epoch+step) % args.tb_interval == 0:
                    with train_writer.as_default():
                        for k, v in metrics.items():
                            tf.summary.scalar(k, v.result(), step=epoch*steps_per_epoch+step)

        if args.tensorboard and args.tb_interval == 0:
            with train_writer.as_default():
                for k, v in metrics.items():
                    tf.summary.scalar(k, v.result(), step=epoch)

        print()
        # save checkpoint, history, and tensorboard
        logs = {k: v.result().numpy() for k, v in metrics.items()}
        logs['epoch'] = epoch + 1

        if args.checkpoint:
            print()
            ckpt_name = '{:04d}_{:.4f}_{:.4f}_{:.4f}'.format(epoch+1, logs['loss'], logs['acc1'], logs['acc5'])
            for n, m in zip(['query', 'key'], [encoder_q, encoder_k]):
                m.save_weights(
                    os.path.join(
                        args.result_path, 
                        f'{args.stamp}/checkpoint/{n}/{ckpt_name}.h5'))
            
                print('Saved at {}'.format(
                        os.path.join(
                            args.result_path, 
                            f'{args.stamp}/checkpoint/{n}/{ckpt_name}.h5')))

            np.save(
                os.path.join(
                    args.result_path, f'{args.stamp}/checkpoint/{ckpt_name}.npy'),
                queue.numpy())
            print()

        if args.history:
            csvlogger = csvlogger.append(logs, ignore_index=True)
            csvlogger.to_csv(os.path.join(args.result_path, 
                                        f'{args.stamp}/history/epoch.csv'), index=False)

        for k, v in metrics.items():
            if not 'total' in k:
                v.reset_states()


if __name__ == "__main__":
    def check_arguments(args):
        assert args.src_path is not None, 'src_path must be entered.'
        assert args.data_path is not None, 'data_path must be entered.'
        assert args.result_path is not None, 'result_path must be entered.'
        return args

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str,       default='v1')
    parser.add_argument("--backbone",       type=str,       default='resnet50')
    parser.add_argument("--batch_size",     type=int,       default=256)
    parser.add_argument("--img_size",       type=int,       default=224)
    parser.add_argument("--dim",            type=int,       default=128)
    parser.add_argument("--num_negative",   type=int,       default=65536)
    parser.add_argument("--momentum",       type=float,     default=.999)
    parser.add_argument("--mlp",            type=int,       default=0) # v2
    parser.add_argument("--shuffle_bn",     action='store_true')
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=200)

    parser.add_argument("--lr",             type=float,     default=.03)
    parser.add_argument("--temperature",    type=float,     default=0.07)

    parser.add_argument("--brightness",     type=float,     default=0.,             help='0.4')
    parser.add_argument("--contrast",       type=float,     default=0.,             help='0.4')
    parser.add_argument("--saturation",     type=float,     default=0.,             help='0.4')
    parser.add_argument("--hue",            type=float,     default=0.,             help='v1: 0.4 / v2: 0.1') # v1 / v2

    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--tb_interval",    type=int,       default=0)
    parser.add_argument("--lr_mode",        type=str,       default='exponential',  choices=['exponential', 'cosine'],
                        help="v1 : exponential | v2 : cosine")
    parser.add_argument("--lr_value",       type=float,     default=.1)
    parser.add_argument("--lr_interval",    type=str,       default='120,160')

    parser.add_argument('--src_path',       type=str,       default='.')
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default='-1')
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--ignore-search",  type=str,       default='')

    args = check_arguments(parser.parse_args())
    main(args)