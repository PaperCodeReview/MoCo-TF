import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
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
    assert args
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
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    num_workers = strategy.num_replicas_in_sync

    BATCH_SIZE = args.batch_size
    assert BATCH_SIZE % strategy.num_replicas_in_sync == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, strategy.num_replicas_in_sync))
    logger.info("GLOBAL BATCH SIZE : {}".format(BATCH_SIZE))
    
    
    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args.data_path, args.dataset)
    steps_per_epoch = args.steps or len(trainset) // BATCH_SIZE
    validation_steps = len(valset) // BATCH_SIZE

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
            'acc'       :   tf.keras.metrics.CategoricalAccuracy('acc', dtype=tf.float32),
            'val_loss'  :   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
            'val_acc'   :   tf.keras.metrics.CategoricalAccuracy('val_acc', dtype=tf.float32),
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
            criterion = tf.losses.sparse_categorical_crossentropy
        else:
            raise ValueError()

        # generator
        train_generator = DataLoader(args, 'train', trainset, BATCH_SIZE, num_workers).dataloader
        val_generator = DataLoader(args, 'val', valset, BATCH_SIZE, num_workers, shuffle=False).dataloader
        
        train_generator = strategy.experimental_distribute_dataset(train_generator)
        val_generator = strategy.experimental_distribute_dataset(val_generator)

    csvlogger, train_writer, val_writer = create_callbacks(args, metrics)
    logger.info("Build Model & Metrics")

    ##########################
    # READY Train
    ##########################
    train_iterator = iter(train_generator)
    val_iterator = iter(val_generator)
    # for t in train_iterator:
    #     print(t[0].shape, t[1].shape)
    
    @tf.function
    def do_step(iterator, mode, queue):
        def step_fn(from_iter):
            img_q, img_k = from_iter['query'], from_iter['key']
            N = tf.shape(img_q)[0]
            K = tf.shape(queue)[0]
            C = tf.shape(queue)[1]

            k = encoder_k(img_k, training=False)
            if mode == 'train':
                with tf.GradientTape() as tape:
                    q = encoder_q(img_q, training=True)
                    l_pos = tf.reshape(
                        tf.matmul(
                            tf.reshape(q, [N, 1, C]), 
                            tf.reshape(k, [N, C, 1])), 
                        [N, 1])
                    l_neg = tf.matmul(
                        tf.reshape(q, [N, C]), 
                        tf.reshape(queue, [C, K]))
                    logits = tf.concat((l_pos, l_neg), axis=1)
                    logits /= args.temperature
                    labels = tf.zeros(N)
                    loss = tf.reduce_mean(criterion(labels, logits))

                grads = tape.gradient(loss, encoder_q.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, encoder_q.trainable_variables)))

                metrics['loss'].update_state(loss)
                metrics['acc'].update_state(labels, logits)

            else:
                q = encoder_q(img_q, training=True)
                l_pos = tf.reshape(
                    tf.matmul(
                        tf.reshape(q, [N, 1, C]), 
                        tf.reshape(k, [N, C, 1])), 
                    [N, 1])
                l_neg = tf.matmul(
                    tf.reshape(q, [N, C]), 
                    tf.reshape(queue, [C, K]))
                logits = tf.concat((l_pos, l_neg), axis=1)
                logits /= args.temperature
                labels = tf.zeros(N)
                loss = tf.reduce_mean(criterion(labels, logits))

                metrics['val_loss'].update_state(loss)
                metrics['val_acc'].update_state(labels, logits)

            return k

        new_key = strategy.run(step_fn, args=(next(iterator),))
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
            for n, m in zip(['query', 'key'], [encoder_q, encoder_k]):
                m.save_weights(
                    os.path.join(
                        args.result_path, 
                        '{}/{}/checkpoint/{}/{:04d}_{:.4f}.h5'.format(
                            args.dataset, args.stamp, n, epoch+1, logs['val_loss'])))
            
                print('\nSaved at {}'.format(
                        os.path.join(
                            args.result_path, 
                            '{}/{}/checkpoint/{}/{:04d}_{:.4f}.h5'.format(
                                args.dataset, args.stamp, n, epoch+1, logs['val_loss']))))

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
    parser.add_argument("--mlp",            action='store_true')
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
    parser.add_argument("--hue",            type=float,     default=0.,             help='v1: 0.4 / v2: 0.1')

    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--lr_mode",        type=str,       default='constant',     choices=['constant', 'exponential', 'cosine'])
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