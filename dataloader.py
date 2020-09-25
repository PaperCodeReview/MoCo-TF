import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from augment import Augment


AUTO = tf.data.experimental.AUTOTUNE


def set_dataset(data_path, dataset):
    trainset = pd.read_csv(
        os.path.join(
            data_path, '{}_trainset.csv'.format(dataset)
        )).values.tolist()
    trainset = [[os.path.join(data_path, t[0]), t[1]] for t in trainset]

    valset = pd.read_csv(
        os.path.join(
            data_path, '{}_valset.csv'.format(dataset)
        )).values.tolist()
    valset = [[os.path.join(data_path, t[0]), t[1]] for t in valset]
    return np.array(trainset, dtype='object'), np.array(valset, dtype='object')


class DataLoader:
    def __init__(self, args, mode, datalist, batch_size, num_workers=1, shuffle=True):
        self.args = args
        self.mode = mode
        self.datalist = datalist
        self.imglist = self.datalist[:,0].tolist()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.dataloader = self._dataloader()

    def __len__(self):
        return len(self.datalist)

    def fetch_dataset(self, path):
        x = tf.io.read_file(path)
        return tf.data.Dataset.from_tensors(x)

    def augmentation(self, img, shape):
        augset = Augment(self.args, self.mode)
        img_list = []
        for _ in range(2): # query, key
            aug_img = tf.identity(img)
            if self.args.task == 'v1':
                aug_img = augset._augmentv1(aug_img, shape) # moco v1
            else:
                aug_img = augset._augmentv2(aug_img, shape) # moco v2
            img_list.append(aug_img)
        return img_list

    def dataset_parser(self, value):
        shape = tf.image.extract_jpeg_shape(value)
        img = tf.io.decode_jpeg(value, channels=3)
        query, key = self.augmentation(img, shape)
        return {'query': query, 'key': key}

    def shuffle_BN(self, value):
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
        return (value, unshuffle_idx)
        
    def _dataloader(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.imglist)
        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(self.datalist))

        dataset = dataset.interleave(self.fetch_dataset, num_parallel_calls=AUTO)
        dataset = dataset.map(self.dataset_parser, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        if self.args.shuffle_bn:
            dataset = dataset.map(self.shuffle_BN, num_parallel_calls=AUTO)
        return dataset