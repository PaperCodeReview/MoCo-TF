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
from model import MoCo
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
    # Model & Generator
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

        Trainer = MoCo(encoder_q, encoder_k, queue)

        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        Trainer.compile(
            optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.0001),
            metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
                     tf.keras.metrics.TopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32)],
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, 
                reduction=tf.keras.losses.Reduction.NONE, name='loss'),
            batch_size=args.batch_size,
            num_negative=args.num_negative,
            temperature=args.temperature,
            momentum=args.momentum,
            shuffle_bn=args.shuffle_bn,
            num_workers=num_workers,
            run_eagerly=True)

    train_generator = DataLoader(args, 'train', trainset, args.batch_size, num_workers).dataloader


    ##########################
    # Train
    ##########################
    callbacks = create_callbacks(args)
    Trainer.fit(
        train_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,)


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
    parser.add_argument("--tb_histogram",   type=int,       default=0)
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