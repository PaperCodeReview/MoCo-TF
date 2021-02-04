import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from augment import Augment


AUTO = tf.data.experimental.AUTOTUNE


def set_dataset(task, data_path):
    trainset = pd.read_csv(
        os.path.join(
            data_path, 'imagenet_trainset.csv'
        )).values.tolist()
    trainset = [[os.path.join(data_path, t[0]), t[1]] for t in trainset]

    if task == 'lincls':
        valset = pd.read_csv(
            os.path.join(
                data_path, 'imagenet_valset.csv'
            )).values.tolist()
        valset = [[os.path.join(data_path, t[0]), t[1]] for t in valset]
        return np.array(trainset, dtype='object'), np.array(valset, dtype='object')

    return np.array(trainset, dtype='object')


class DataLoader:
    def __init__(self, args, mode, datalist, batch_size, num_workers=1, shuffle=True):
        self.args = args
        self.mode = mode
        self.datalist = datalist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.dataloader = self._dataloader()

    def __len__(self):
        return len(self.datalist)

    def fetch_dataset(self, path, y=None):
        x = tf.io.read_file(path)
        if y is not None:
            return tf.data.Dataset.from_tensors((x, y))
        return tf.data.Dataset.from_tensors(x)

    def augmentation(self, img, shape):
        augset = Augment(self.args, self.mode)
        if self.args.task in ['v1', 'v2']:
            img_list = []
            for _ in range(2): # query, key
                aug_img = tf.identity(img)
                if self.args.task == 'v1':
                    aug_img = augset._augmentv1(aug_img, shape) # moco v1
                else:
                    radius = np.random.choice([3, 5])
                    aug_img = augset._augmentv2(aug_img, shape, (radius, radius)) # moco v2
                img_list.append(aug_img)
            return img_list
        else:
            return augset._augment_lincls(img, shape)

    def dataset_parser(self, value, label=None):
        shape = tf.image.extract_jpeg_shape(value)
        img = tf.io.decode_jpeg(value, channels=3)
        if label is None:
            # moco
            query, key = self.augmentation(img, shape)
            inputs = {'query': query, 'key': key}
            labels = tf.zeros([])
        else:
            # lincls
            inputs = self.augmentation(img, shape)
            labels = tf.one_hot(label, self.args.classes)
        return (inputs, labels)

    def shuffle_BN(self, value, labels):
        if self.num_workers > 1:
            pre_shuffle = [(i, value['key'][i]) for i in range(self.batch_size)]
            random.shuffle(pre_shuffle)
            shuffle_idx = []
            value_temp = []
            for vv in pre_shuffle:
                shuffle_idx.append(vv[0])
                value_temp.append(tf.expand_dims(vv[1], axis=0))
            value['key'] = tf.concat(value_temp, axis=0)
            unshuffle_idx = np.array(shuffle_idx).argsort().tolist()
            value.update({'unshuffle': unshuffle_idx})
        return (value, labels)
        
    def _dataloader(self):
        self.imglist = self.datalist[:,0].tolist()
        if self.args.task in ['v1', 'v2']:
            dataset = tf.data.Dataset.from_tensor_slices(self.imglist)
        else:
            self.labellist = self.datalist[:,1].tolist()
            dataset = tf.data.Dataset.from_tensor_slices((self.imglist, self.labellist))

        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(self.datalist))

        dataset = dataset.interleave(self.fetch_dataset, num_parallel_calls=AUTO)
        dataset = dataset.map(self.dataset_parser, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        if self.args.shuffle_bn and self.args.task in ['v1', 'v2']:
            # only moco
            dataset = dataset.map(self.shuffle_BN, num_parallel_calls=AUTO)
        return dataset