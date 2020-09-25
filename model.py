import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model


model_dict = {
    'vgg16'         : tf.keras.applications.VGG16,
    'vgg19'         : tf.keras.applications.VGG19,
    'resnet50'      : tf.keras.applications.ResNet50,
    'resnet50v2'    : tf.keras.applications.ResNet50V2,
    'resnet101'     : tf.keras.applications.ResNet101,
    'resnet101v2'   : tf.keras.applications.ResNet101V2,
    'resnet152'     : tf.keras.applications.ResNet152,
    'resnet152v2'   : tf.keras.applications.ResNet152V2,
    'xception'      : tf.keras.applications.Xception, # 299
    'densenet121'   : tf.keras.applications.DenseNet121, # 224
    'densenet169'   : tf.keras.applications.DenseNet169, # 224
    'densenet201'   : tf.keras.applications.DenseNet201, # 224
}
   

def create_model(
    logger, 
    backbone='resnet50',
    img_size=224,
    dim=128, 
    K=65536,
    mlp=False,
    snapshot=None):

    def _get_architecture(name=None):
        base_encoder = model_dict[backbone](
            include_top=False,
            pooling='avg',
            weights=None,
            input_shape=(img_size, img_size, 3))

        x = Dense(dim)(base_encoder.output)
        if mlp:
            x = Activation('relu')(x)
            x = Dense(dim)(x)
        arch = Model(base_encoder.input, x, name=name)
        return arch
    
    encoder_q = _get_architecture('encoder_q_{}'.format(backbone))
    encoder_k = _get_architecture('encoder_k_{}'.format(backbone))

    for i in range(len(encoder_q.layers)):
        encoder_k.get_layer(index=i).set_weights(
            encoder_q.get_layer(index=i).get_weights())
    encoder_k.trainable = False

    if snapshot:
        encoder_q.load_weights(snapshot)
        logger.info('Load query weights at {}'.format(snapshot))
        encoder_k.load_weights(snapshot.replace('/query/', '/key/'))
        logger.info('Load key weights at {}'.format(snapshot.replace('/query/', '/key/')))
        queue = tf.constant(np.load(snapshot.replace('/query', '').replace('.h5', '.npy')), dtype=tf.float32)
        logger.info('Load queue at {}'.format(snapshot.replace('/query', '').replace('.h5', '.npy')))
    else:
        # queue
        queue = tf.random.normal(shape=[K, dim])
        queue /= tf.norm(queue, ord=2, axis=0)
    return encoder_q, encoder_k, queue


def momentum_update_model(encoder_q, encoder_k, m=0.999):
    for i in range(len(encoder_q.layers)):
        param_q = encoder_q.get_layer(index=i).get_weights()
        param_k = encoder_k.get_layer(index=i).get_weights()
        encoder_k.get_layer(index=i).set_weights(
            [param_k[j] * m + param_q[j] * (1. - m) for j in range(len(param_q))])
    return encoder_k


def enqueue(queue, new_keys, K=65536):
    queue = tf.concat([new_keys, queue], axis=0)
    queue = queue[:K]
    return queue