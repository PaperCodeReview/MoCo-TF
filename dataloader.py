import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

mean_std = {
    'cub': [[0.48552202, 0.49934904, 0.43224954], 
            [0.18172876, 0.18109447, 0.19272076]],
    'cifar100': [[0.50707516, 0.48654887, 0.44091784], 
                 [0.20079844, 0.19834627, 0.20219835]],
}

def set_dataset(args):
    trainset = pd.read_csv(
        os.path.join(
            args.data_path, '{}_trainset.csv'.format(args.dataset)
        )).values.tolist()
    valset = pd.read_csv(
        os.path.join(
            args.data_path, '{}_valset.csv'.format(args.dataset)
        )).values.tolist()
    return np.array(trainset, dtype='object'), np.array(valset, dtype='object')