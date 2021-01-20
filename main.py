import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from common import set_seed
from common import get_logger
from common import get_session
from common import get_arguments
from common import search_same
from common import create_stamp
from dataloader import set_dataset
from dataloader import DataLoader
from model import MoCo
from model import set_lincls
from callback import OptionalLearningRateSchedule
from callback import create_callbacks

import tensorflow as tf


def train_moco(args, logger, initial_epoch, strategy, num_workers):
    ##########################
    # Dataset
    ##########################
    trainset = set_dataset(args.task, args.data_path)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== TRAINSET ==========")
    logger.info(f"    --> {len(trainset)}")
    logger.info(f"    --> {steps_per_epoch}")


    ##########################
    # Model & Generator
    ##########################
    with strategy.scope():
        model = MoCo(args, logger)

        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
                     tf.keras.metrics.TopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32)],
            num_workers=num_workers,
            run_eagerly=True)
    
    train_generator = DataLoader(args, 'train', trainset, args.batch_size, num_workers).dataloader

    ##########################
    # Train
    ##########################
    callbacks, initial_epoch = create_callbacks(args, logger, initial_epoch)
    if callbacks == -1:
        logger.info('Check your model.')
        return
    elif callbacks == -2:
        return

    model.fit(
        train_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,)


def train_lincls(args, logger, initial_epoch, strategy, num_workers):
    assert args.snapshot is not None, 'pretrained weight is needed!'
    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args.task, args.data_path)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    validation_steps = len(valset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== TRAINSET ==========")
    logger.info(f"    --> {len(trainset)}")
    logger.info(f"    --> {steps_per_epoch}")

    logger.info("=========== VALSET ===========")
    logger.info(f"    --> {len(valset)}")
    logger.info(f"    --> {validation_steps}")


    ##########################
    # Model & Generator
    ##########################
    with strategy.scope():
        backbone = MoCo(args, logger)
        model = set_lincls(args, backbone.encoder_q)

        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9),
            metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
                     tf.keras.metrics.TopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32)],
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='loss'))

    train_generator = DataLoader(args, 'train', trainset, args.batch_size, num_workers).dataloader
    val_generator = DataLoader(args, 'val', valset, args.batch_size, num_workers).dataloader


    ##########################
    # Train
    ##########################
    callbacks, initial_epoch = create_callbacks(args, logger, initial_epoch)
    if callbacks == -1:
        logger.info('Check your model.')
        return
    elif callbacks == -2:
        return

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps)


def main():
    set_seed()
    args = get_arguments()
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
    if len(args.gpus.split(',')) > 1:
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    num_workers = strategy.num_replicas_in_sync
    assert args.batch_size % num_workers == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, num_workers))
    logger.info("GLOBAL BATCH SIZE : {}".format(args.batch_size))


    ##########################
    # Training
    ##########################
    if args.task in ['v1', 'v2']:
        train_moco(args, logger, initial_epoch, strategy, num_workers)
    else:
        train_lincls(args, logger, initial_epoch, strategy, num_workers)
    
    
if __name__ == "__main__":
    main()