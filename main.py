import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import argparse
from datetime import datetime

import tensorflow as tf

from dataloader import set_dataset
from loss import crossentropy


def main(args):
    sys.path.append(args.baseline_path)
    from common import get_logger
    from common import get_session
    from common import search_same
    from callback_eager import OptionalLearningRateSchedule
    from callback_eager import create_callbacks

    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        temp = datetime.now()
        args.stamp = "{:02d}{:02d}{:02d}_{}_{:02d}_{:02d}_{:02d}".format(
            temp.year // 100,
            temp.month,
            temp.day,
            weekday[temp.weekday()],
            temp.hour,
            temp.minute,
            temp.second,
        )

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args)

    ##########################
    # Model & Metric & Generator
    ##########################
    progress_desc_train = 'Train : Loss {:.4f} | Acc {:.4f}'
    progress_desc_val = 'Val : Loss {:.4f} | Acc {:.4f}'

    # select your favorite distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    
    steps_per_epoch = args.steps or len(trainset) // global_batch_size
    validation_steps = len(valset) // global_batch_size

    # lr scheduler
    lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)

    with strategy.scope():
        model = create_model(args, logger)
        if args.summary:
            model.summary()
            return

        # metrics
        metrics = {
            'loss'      :   tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'acc'       :   tf.keras.metrics.CategoricalAccuracy('acc', dtype=tf.float32),
            'val_loss'  :   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
            'val_acc'   :   tf.keras.metrics.CategoricalAccuracy('val_acc', dtype=tf.float32),
        }

        # optimizer
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.0001)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
        elif args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        # loss
        if args.loss == 'crossentropy':
            criterion = crossentropy(args)
        else:
            raise ValueError()

        # generator
        if args.loss == 'crossentropy':
            train_generator = dataloader(args, trainset, 'train', global_batch_size)
            val_generator = dataloader(args, valset, 'val', global_batch_size, shuffle=False)
        else:
            raise ValueError()
        
        train_generator = strategy.experimental_distribute_dataset(train_generator)
        val_generator = strategy.experimental_distribute_dataset(val_generator)

    path = os.path.join(args.result_path, args.dataset, args.model_name, str(args.stamp))
    csvlogger, train_writer, val_writer = create_callbacks(args, metrics, path)
    logger.info("Build Model & Metrics")

    ##########################
    # Log Arguments & Settings
    ##########################
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    logger.info('{} : {}'.format(strategy.__class__.__name__, strategy.num_replicas_in_sync))
    logger.info("GLOBAL BATCH SIZE : {}".format(global_batch_size))

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))

    ##########################
    # READY Train
    ##########################
    train_iterator = iter(train_generator)
    val_iterator = iter(val_generator)
        
    @tf.function
    def do_step(iterator, mode, loss_name, acc_name=None):
        def step_fn(from_iterator):
            inputs, labels = from_iterator
            if mode == 'train':
                # TODO : loss 계산 다시하기
                with tf.GradientTape() as tape:
                    logits = tf.cast(model(inputs, training=True), tf.float32)
                    loss = criterion(labels, logits)
                    loss = tf.reduce_sum(loss) * (1./global_batch_size)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            else:
                logits = tf.cast(model(inputs, training=False), tf.float32)
                loss = criterion(labels, logits)
                loss = tf.reduce_sum(loss) * (1./global_batch_size)

            metrics[loss_name].update_state(loss)
            metrics[acc_name].update_state(labels, logits)

        strategy.run(step_fn, args=(next(iterator),))
        # step_fn(next(iterator))

    def desc_update(pbar, desc, loss, acc=None):
        pbar.set_description(desc.format(loss.result(), acc.result()))

    ##########################
    # Train
    ##########################
    for epoch in range(initial_epoch, args.epochs):
        print('\nEpoch {}/{}'.format(epoch+1, args.epochs))
        print('Learning Rate : {}'.format(optimizer.learning_rate(optimizer.iterations)))

        # train
        progressbar_train = tqdm.tqdm(
            tf.range(steps_per_epoch), 
            desc=progress_desc_train.format(0, 0, 0, 0), 
            leave=True)
        for step in progressbar_train:
            do_step(train_iterator, 'train', 'loss', 'acc')
            desc_update(progressbar_train, progress_desc_train, metrics['loss'], metrics['acc'])
            progressbar_train.refresh()

        # eval
        progressbar_val = tqdm.tqdm(
            tf.range(validation_steps), 
            desc=progress_desc_val.format(0, 0), 
            leave=True)
        for step in progressbar_val:
            do_step(val_iterator, 'val', 'val_loss', 'val_acc')
            desc_update(progressbar_val, progress_desc_val, metrics['val_loss'], metrics['val_acc'])
            progressbar_val.refresh()
    
        # logs
        logs = {k: v.result().numpy() for k, v in metrics.items()}
        logs['epoch'] = epoch

        if args.checkpoint:
            ckpt_path = '{:04d}_{:.4f}_{:.4f}.h5'.format(epoch+1, logs['val_acc'], logs['val_loss'])
            model.save_weights(os.path.join(path, 'checkpoint', ckpt_path))
            print('\nSaved at {}'.format(os.path.join(path, 'checkpoint', ckpt_path)))

        if args.history:
            csvlogger = csvlogger.append(logs, ignore_index=True)
            csvlogger.to_csv(os.path.join(path, 'history/epoch.csv'), index=False)

        if args.tensorboard:
            with train_writer.as_default():
                tf.summary.scalar('loss', metrics['loss'].result(), step=epoch)
                tf.summary.scalar('acc', metrics['acc'].result(), step=epoch)

            with val_writer.as_default():
                tf.summary.scalar('val_loss', metrics['val_loss'].result(), step=epoch)
                tf.summary.scalar('val_acc', metrics['val_acc'].result(), step=epoch)
        
        for k, v in metrics.items():
            v.reset_states()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",       type=str,       default='resnet50')
    parser.add_argument("--batch-size",     type=int,       default=32,
                        help="batch size per replica")
    parser.add_argument("--classes",        type=int,       default=200)
    parser.add_argument("--dataset",        type=str,       default='imagenet')
    parser.add_argument("--img-size",       type=int,       default=224)
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=100)

    parser.add_argument("--optimizer",      type=str,       default='sgd')
    parser.add_argument("--lr",             type=float,     default=.001)
    parser.add_argument("--loss",           type=str,       default='crossentropy', choices=['crossentropy', 'supcon'])
    parser.add_argument("--temperature",    type=float,     default=0.07)
    parser.add_argument("--momentum",       type=float,     default=0.999)

    parser.add_argument("--augment",        type=str,       default='sim')
    parser.add_argument("--standardize",    type=str,       default='minmax1',      choices=['minmax1', 'minmax2', 'norm', 'eachnorm'])

    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--lr-mode",        type=str,       default='constant',     choices=['constant', 'exponential', 'cosine'])
    parser.add_argument("--lr-value",       type=float,     default=.1)
    parser.add_argument("--lr-interval",    type=str,       default='20,50,80')
    parser.add_argument("--lr-warmup",      type=int,       default=0)

    parser.add_argument('--baseline-path',  type=str,       default='/workspace/src/Challenge/code_baseline')
    parser.add_argument('--src-path',       type=str,       default='.')
    parser.add_argument('--data-path',      type=str,       default=None)
    parser.add_argument('--result-path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default='-1')
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--ignore-search",  type=str,       default='')

    main(parser.parse_args())