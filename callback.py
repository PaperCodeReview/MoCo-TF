import os
import yaml
import pandas as pd
import tensorflow as tf

from common import create_stamp


class OptionalLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, args, steps_per_epoch, initial_epoch):
        super(OptionalLearningRateSchedule, self).__init__()
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.initial_epoch = initial_epoch

        if self.args.lr_mode == 'exponential':
            decay_epochs = [int(e) for e in self.args.lr_interval.split(',')]
            lr_values = [self.args.lr * (self.args.lr_value ** k)for k in range(len(decay_epochs) + 1)]
            self.lr_scheduler = \
                tf.keras.optimizers.schedules.PiecewiseConstantDecay(decay_epochs, lr_values)

        elif self.args.lr_mode == 'cosine':
            self.lr_scheduler = \
                tf.keras.experimental.CosineDecay(self.args.lr, self.args.epochs-self.args.lr_warmup)

        elif self.args.lr_mode == 'constant':
            self.lr_scheduler = lambda x: self.args.lr
            

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'init_lr': self.args.lr,
            'lr_mode': self.args.lr_mode,
            'lr_warmup': self.args.lr_warmup,
            'lr_value': self.args.lr_value,
            'lr_interval': self.args.lr_interval,
        }

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step += self.initial_epoch * self.steps_per_epoch
        lr_epoch = (step / self.steps_per_epoch)
        if self.args.lr_warmup:
            total_lr_warmup_step = self.args.lr_warmup * self.steps_per_epoch
            return tf.cond(lr_epoch < self.args.lr_warmup,
                           lambda: step / total_lr_warmup_step * self.args.lr,
                           lambda: self.lr_scheduler(lr_epoch-self.args.lr_warmup))
        else:
            if self.args.lr_mode == 'constant':
                return self.args.lr
            else:
                return self.lr_scheduler(lr_epoch)


def create_callbacks(args, metrics):
    if args.snapshot is None:
        if args.checkpoint or args.history or args.tensorboard:
            flag = True
            while flag:
                try:
                    os.makedirs(os.path.join(args.result_path, args.dataset, args.stamp))
                    flag = False
                except:
                    args.stamp = create_stamp()

            yaml.dump(
                vars(args), 
                open(os.path.join(args.result_path, args.dataset, args.stamp, "model_desc.yml"), "w"), 
                default_flow_style=False)

    if args.checkpoint:
        os.makedirs(os.path.join(args.result_path, '{}/{}/checkpoint/query'.format(args.dataset, args.stamp)), exist_ok=True)
        os.makedirs(os.path.join(args.result_path, '{}/{}/checkpoint/key'.format(args.dataset, args.stamp)), exist_ok=True)

    if args.history:
        os.makedirs(os.path.join(args.result_path, '{}/{}/history'.format(args.dataset, args.stamp)), exist_ok=True)
        csvlogger = pd.DataFrame(columns=['epoch']+list(metrics.keys()))
        if os.path.isfile(os.path.join(args.result_path, '{}/{}/history/epoch.csv'.format(args.dataset, args.stamp))):
            csvlogger = pd.read_csv(os.path.join(args.result_path, '{}/{}/history/epoch.csv'.format(args.dataset, args.stamp)))
        else:
            csvlogger.to_csv(os.path.join(args.result_path, '{}/{}/history/epoch.csv'.format(args.dataset, args.stamp)), index=False)
    else:
        csvlogger = None

    if args.tensorboard:
        train_writer = tf.summary.create_file_writer(os.path.join(args.result_path, args.dataset, args.stamp, 'logs/train'))
        val_writer = tf.summary.create_file_writer(os.path.join(args.result_path, args.dataset, args.stamp, 'logs/val'))
    else:
        train_writer = val_writer = None

    return csvlogger, train_writer, val_writer