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


class MoCo(tf.keras.Model):
    def __init__(self, encoder_q, encoder_k, queue):
        super(MoCo, self).__init__()
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.queue = queue

    def compile(
        self,
        optimizer,
        metrics,
        loss,
        batch_size=256,
        num_negative=65536,
        temperature=.07,
        momentum=.999,
        shuffle_bn=False,
        num_workers=1,
        run_eagerly=None):

        super(MoCo, self).compile(optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly)
        self.loss = loss
        self.batch_size = batch_size
        self.num_negative = num_negative
        self.temperature = temperature
        self.momentum = momentum
        self.shuffle_bn = shuffle_bn
        self.num_workers = num_workers

    def train_step(self, data):
        if self.shuffle_bn:
            x, unshuffle_idx = data
            img_q, img_k = x['query'], x['key']
        else:
            img_q, img_k = data['query'], data['key']

        N = tf.shape(img_q)[0]
        C = tf.shape(self.queue)[0]
        K = tf.shape(self.queue)[1]
        labels = tf.one_hot(tf.zeros(N, dtype=tf.int32), self.num_negative+1) # labels : (256, 65536+1)

        key = self.encoder_k(img_k, training=False)
        key = tf.math.l2_normalize(key, axis=1)
        if self.shuffle_bn and self.num_workers > 1:
            def concat_fn(strategy, key_per_replica):
                return tf.concat(key_per_replica.values, axis=0)
            replica_context = tf.distribute.get_replica_context()
            key_all_replica = replica_context.merge_call(concat_fn, args=(key,))
            unshuffle_idx_all_replica = replica_context.merge_call(concat_fn, args=(unshuffle_idx,))
            new_key_list = []
            for idx in unshuffle_idx_all_replica:
                new_key_list.append(tf.expand_dims(key_all_replica[idx], axis=0))
            key_orig = tf.concat(tuple(new_key_list), axis=0)
            key = key_orig[(self.batch_size//self.num_workers)*(replica_context.replica_id_in_sync_group):
                            (self.batch_size//self.num_workers)*(replica_context.replica_id_in_sync_group+1)]

        with tf.GradientTape() as tape:
            query = self.encoder_q(img_q, training=True)
            query = tf.math.l2_normalize(query, axis=1)
            l_pos = tf.reshape(
                tf.matmul(
                    tf.reshape(query, [N, 1, C]), 
                    tf.reshape(tf.stop_gradient(key), [N, C, 1])), 
                [N, 1])                                 # l_pos : (256, 1)
            l_neg = tf.matmul(query, self.queue)        # l_neg : (256, 65536)
            logits = tf.concat((l_pos, l_neg), axis=1)  # logits : (256, 65536+1)
            logits /= self.temperature
            loss = self.loss(labels, logits)
            loss = tf.reduce_sum(loss) / self.batch_size

        trainable_vars = self.encoder_q.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(labels, logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})

        self.enqueue(key)
        self.momentum_update_model()
        return results

    def enqueue(self, new_keys):
        self.queue = tf.concat([tf.transpose(new_keys), self.queue], axis=1)
        self.queue = self.queue[:,:self.num_negative]

    def momentum_update_model(self):
        for i in range(len(self.encoder_q.layers)):
            param_q = self.encoder_q.get_layer(index=i).get_weights()
            param_k = self.encoder_k.get_layer(index=i).get_weights()
            self.encoder_k.get_layer(index=i).set_weights(
                [param_k[j] * self.momentum + param_q[j] * (1. - self.momentum) 
                 for j in range(len(param_q))])
   

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
        x = base_encoder.output
        x = Dense(dim)(x)
        if mlp:
            x = Activation('relu')(x)
            x = Dense(dim)(x)
        arch = Model(base_encoder.input, x, name=name)
        return arch
    
    encoder_q = _get_architecture('encoder_q_{}'.format(backbone))
    encoder_k = _get_architecture('encoder_k_{}'.format(backbone))

    if snapshot:
        encoder_q.load_weights(snapshot)
        logger.info('Load query weights at {}'.format(snapshot))
        encoder_k.load_weights(snapshot.replace('/query/', '/key/'))
        logger.info('Load key weights at {}'.format(snapshot.replace('/query/', '/key/')))
        queue = tf.constant(np.load(snapshot.replace('/query/', '/queue/').replace('.h5', '.npy')), dtype=tf.float32)
        logger.info('Load queue at {}'.format(snapshot.replace('/query/', '/queue/').replace('.h5', '.npy')))
    else:
        for i in range(len(encoder_q.layers)):
            encoder_k.get_layer(index=i).set_weights(
                encoder_q.get_layer(index=i).get_weights())

        queue = tf.random.normal(shape=[dim, K])
        queue = tf.math.l2_normalize(queue, axis=0)

    encoder_k.trainable = False
    return encoder_q, encoder_k, queue