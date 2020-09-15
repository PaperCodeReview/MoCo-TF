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
    # strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    num_workers = strategy.num_replicas_in_sync
    assert args.batch_size % strategy.num_replicas_in_sync == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, strategy.num_replicas_in_sync))
    logger.info("GLOBAL BATCH SIZE : {}".format(args.batch_size))
    
    
    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args.data_path, args.dataset)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    validation_steps = len(valset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))


    ##########################
    # Model & Metric & Generator
    ##########################
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

        # metrics
        metrics = {
            'loss'      :   tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'acc1'      :   tf.keras.metrics.TopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
            'acc5'      :   tf.keras.metrics.TopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32),
            'val_loss'  :   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
            'val_acc1'  :   tf.keras.metrics.TopKCategoricalAccuracy(1, 'val_acc1', dtype=tf.float32),
            'val_acc5'  :   tf.keras.metrics.TopKCategoricalAccuracy(5, 'val_acc5', dtype=tf.float32),
        }

        # optimizer
        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.0001)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
        elif args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        # loss
        if args.loss == 'crossentropy':
            criterion = tf.keras.losses.categorical_crossentropy
        else:
            raise ValueError()

        # generator
        train_generator = DataLoader(args, 'train', trainset, args.batch_size, num_workers).dataloader
        val_generator = DataLoader(args, 'val', valset, args.batch_size, num_workers, shuffle=False).dataloader
        # for t in train_generator:
        #     print(t['query'].shape, tf.reduce_min(t['query']), tf.reduce_max(t['query']), t['key'].shape)
        
        train_generator = strategy.experimental_distribute_dataset(train_generator)
        val_generator = strategy.experimental_distribute_dataset(val_generator)

    csvlogger, train_writer, val_writer = create_callbacks(args, metrics)
    logger.info("Build Model & Metrics")

    ##########################
    # READY Train
    ##########################
    train_iterator = iter(train_generator)
    val_iterator = iter(val_generator)
    
    @tf.function
    def do_step(iterator, mode, queue):
        def get_loss(img_q, key, labels, N, C, K):
            query = encoder_q(img_q, training=True)
            l_pos = tf.reshape(
                tf.matmul(
                    tf.reshape(query, [N, 1, C]), 
                    tf.reshape(key, [N, C, 1])), 
                [N, 1]) # (N, 1)
            l_neg = tf.matmul(
                tf.reshape(query, [N, C]), 
                tf.reshape(queue, [C, K])) # (N, K)
            logits = tf.concat((l_pos, l_neg), axis=1) # (N, K+1)
            logits /= args.temperature
            loss = criterion(labels, logits, from_logits=True)
            loss_mean = tf.nn.compute_average_loss(loss, global_batch_size=args.batch_size)
            return logits, loss, loss_mean

        def step_fn(from_iter):
            if args.shuffle_bn:
                x, unshuffle_idx = from_iter
                img_q, img_k = x['query'], x['key']
            else:
                img_q, img_k = from_iter['query'], from_iter['key']

            N = tf.shape(img_q)[0]
            K = tf.shape(queue)[0]
            C = tf.shape(queue)[1]
            labels = tf.one_hot(tf.zeros(N, dtype=tf.int32), args.num_negative+1)

            key = encoder_k(img_k, training=False)
            ##########################
            # TODO : Shuffling BN
            ##########################
            if args.shuffle_bn:
                raise NotImplementedError()

            if mode == 'train':
                with tf.GradientTape() as tape:
                    logits, loss, loss_mean = get_loss(img_q, key, labels, N, C, K)
                
                grads = tape.gradient(loss_mean, encoder_q.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, encoder_q.trainable_variables)))

            else:
                logits, loss, loss_mean = get_loss(img_q, key, labels, N, C, K)

            metrics['acc1' if mode == 'train' else 'val_acc1'].update_state(labels, logits)
            metrics['acc5' if mode == 'train' else 'val_acc5'].update_state(labels, logits)
            return key, loss

        new_key, loss_per_replica = strategy.run(step_fn, args=(next(iterator),))
        loss_mean = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_replica, axis=0)
        metrics['loss' if mode == 'train' else 'val_loss'].update_state(loss_mean)
        if mode == 'train':
            queue = enqueue(queue, tf.concat(new_key.values, axis=0), K=args.num_negative)
        return queue


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
            queue = do_step(train_iterator, 'train', queue)
            encoder_k = momentum_update_model(encoder_q, encoder_k, m=args.momentum)
            progBar_train.update(step, values=[(k, v.result()) for k, v in metrics.items() if not 'val' in k])
            
            if args.tensorboard and args.tb_interval > 0:
                if (epoch*steps_per_epoch+step) % args.tb_interval == 0:
                    with train_writer.as_default():
                        for k, v in metrics.items():
                            if not 'val' in k:
                                tf.summary.scalar(k, v.result(), step=epoch*steps_per_epoch+step)

        if args.tensorboard and args.tb_interval == 0:
            with train_writer.as_default():
                for k, v in metrics.items():
                    if not 'val' in k:
                        tf.summary.scalar(k, v.result(), step=epoch)

        # val
        print('\n\nValidation')
        encoder_q.trainable = False
        progBar_val = tf.keras.utils.Progbar(validation_steps, stateful_metrics=metrics.keys())
        for step in range(validation_steps):
            do_step(val_iterator, 'val', queue)
            progBar_val.update(step, values=[(k, v.result()) for k, v in metrics.items() if 'val' in k])

        if args.tensorboard:
            with val_writer.as_default():
                for k, v in metrics.items():
                    if 'val' in k:
                        tf.summary.scalar(k.split('val_')[1], v.result(), 
                                        step=epoch if args.tb_interval == 0 else epoch*steps_per_epoch+step)
        print()
        # save checkpoint, history, and tensorboard
        logs = {k: v.result().numpy() for k, v in metrics.items()}
        logs['epoch'] = epoch + 1

        if args.checkpoint:
            print()
            for n, m in zip(['query', 'key'], [encoder_q, encoder_k]):
                m.save_weights(
                    os.path.join(
                        args.result_path, 
                        '{}/{}/checkpoint/{}/{:04d}_{:.4f}_{:.4f}_{:.4f}.h5'.format(
                            args.dataset, args.stamp, n, epoch+1, logs['val_loss'], logs['val_acc1'], logs['val_acc5'])))
            
                print('Saved at {}'.format(
                        os.path.join(
                            args.result_path, 
                            '{}/{}/checkpoint/{}/{:04d}_{:.4f}_{:.4f}_{:.4f}.h5'.format(
                                args.dataset, args.stamp, n, epoch+1, logs['val_loss'], logs['val_acc1'], logs['val_acc5']))))

            np.save(
                os.path.join(
                    args.result_path, '{}/{}/checkpoint/{:04d}_{:.4f}_{:.4f}_{:.4f}.npy'.format(
                        args.dataset, args.stamp, epoch+1, logs['val_loss'], logs['val_acc1'], logs['val_acc5'])),
                queue.numpy())
            print()

        if args.history:
            csvlogger = csvlogger.append(logs, ignore_index=True)
            csvlogger.to_csv(os.path.join(args.result_path, 
                                        f'{args.dataset}/{args.stamp}/history/epoch.csv'), index=False)

        for k, v in metrics.items():
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
    parser.add_argument("--dataset",        type=str,       default='cub')
    parser.add_argument("--img_size",       type=int,       default=224)
    parser.add_argument("--dim",            type=int,       default=128)
    parser.add_argument("--num_negative",   type=int,       default=65536)
    parser.add_argument("--momentum",       type=float,     default=.999)
    parser.add_argument("--mlp",            action='store_true') # v2
    parser.add_argument("--shuffle_bn",     action='store_true')
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=100)

    parser.add_argument("--optimizer",      type=str,       default='sgd')
    parser.add_argument("--lr",             type=float,     default=.03)
    parser.add_argument("--loss",           type=str,       default='crossentropy', choices=['crossentropy'])
    parser.add_argument("--temperature",    type=float,     default=0.07)

    parser.add_argument("--standardize",    type=str,       default='norm',         choices=['minmax1', 'minmax2', 'norm', 'eachnorm'])
    parser.add_argument("--brightness",     type=float,     default=0.,             help='0.4')
    parser.add_argument("--contrast",       type=float,     default=0.,             help='0.4')
    parser.add_argument("--saturation",     type=float,     default=0.,             help='0.4')
    parser.add_argument("--hue",            type=float,     default=0.,             help='v1: 0.4 / v2: 0.1') # v1 / v2

    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--tb_interval",    type=int,       default=0)
    parser.add_argument("--lr_mode",        type=str,       default='constant',     choices=['constant', 'exponential', 'cosine']) # cosine is for v2
    parser.add_argument("--lr_value",       type=float,     default=.1)
    parser.add_argument("--lr_interval",    type=str,       default='20,50,80')
    parser.add_argument("--lr_warmup",      type=int,       default=0)

    parser.add_argument('--src_path',       type=str,       default='.')
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default='-1')
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--ignore-search",  type=str,       default='')

    args = check_arguments(parser.parse_args())
    main(args)