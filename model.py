import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


WEIGHTS_HASHES = {'resnet50' : '4d473c1dd8becc155b73f8504c6f6626',}
MODEL_DICT = {'resnet50' : tf.keras.applications.ResNet50,}
FAMILY_DICT = {'resnet50' : tf.python.keras.applications.resnet,}


def _conv2d(**custom_kwargs):
    def _func(*args, **kwargs):
        kwargs.update(**custom_kwargs)
        return Conv2D(*args, **kwargs)
    return _func


def _dense(**custom_kwargs):
    def _func(*args, **kwargs):
        kwargs.update(**custom_kwargs)
        return Dense(*args, **kwargs)
    return _func


def set_lincls(args, backbone):
    DEFAULT_ARGS = {
        "use_bias": args.use_bias,
        "kernel_regularizer": l2(args.weight_decay)}
    
    if args.freeze:
        backbone.trainable = False
        
    x = backbone.get_layer(name='avg_pool').output
    x = _dense(**DEFAULT_ARGS)(args.classes, name='predictions')(x)
    model = Model(backbone.input, x, name='lincls')
    return model


class MoCo(Model):
    def __init__(self, args, logger, **kwargs):
        super(MoCo, self).__init__(**kwargs)
        self.args = args
        
        DEFAULT_ARGS = {
            "use_bias": self.args.use_bias,
            "kernel_regularizer": l2(self.args.weight_decay)}
        FAMILY_DICT[self.args.backbone].Conv2D = _conv2d(**DEFAULT_ARGS)
        FAMILY_DICT[self.args.backbone].Dense = _dense(**DEFAULT_ARGS)

        def set_encoder(name):
            backbone = MODEL_DICT[self.args.backbone](
                include_top=False,
                weights=None,
                input_shape=(self.args.img_size, self.args.img_size, 3),
                pooling='avg')
            
            x = backbone.output
            x = _dense(**DEFAULT_ARGS)(self.args.dim, name='proj_fc1')(x)
            if args.mlp:
                x = Activation('relu', name='proj_relu1')(x)
                x = _dense(**DEFAULT_ARGS)(self.args.dim, name='proj_fc2')(x)
            encoder = Model(backbone.input, x, name=name)
            return encoder
        
        logger.info('Set query encoder')
        self.encoder_q = set_encoder(name='encoder_q')
        logger.info('Set key encoder')
        self.encoder_k = set_encoder(name='encoder_k')
        logger.info('Set queue')
        _queue = np.random.normal(size=(self.args.dim, self.args.num_negative))
        _queue /= np.linalg.norm(_queue, axis=0)
        self.queue = self.add_weight(
            name='queue',
            shape=(self.args.dim, self.args.num_negative),
            initializer=tf.keras.initializers.Constant(_queue),
            trainable=False)

        if self.args.snapshot:
            self.load_weights(self.args.snapshot)
            logger.info('Load weights at {}'.format(self.args.snapshot))
        else:
            for i in range(len(self.encoder_q.layers)):
                self.encoder_k.get_layer(index=i).set_weights(
                    self.encoder_q.get_layer(index=i).get_weights())
                    
        self.encoder_k.trainable = False

    def compile(
        self,
        optimizer,
        loss,
        metrics,
        num_workers=1,
        run_eagerly=None):

        super(MoCo, self).compile(
            optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly)

        self._loss = loss
        self._num_workers = num_workers
        self._is_shufflebn = self.args.shuffle_bn and self._num_workers > 1
        self._replica_context = tf.distribute.get_replica_context()

    def train_step(self, data):
        inputs, labels = data
        img_q, img_k = inputs['query'], inputs['key']
        if self._is_shufflebn:
            unshuffle_idx = inputs['unshuffle']

        key = tf.cast(self.encoder_k(img_k, training=False), tf.float32)
        key = tf.math.l2_normalize(key, axis=1)
        if self._is_shufflebn:
            key = self.unshuffle_bn(key, unshuffle_idx)

        with tf.GradientTape() as tape:
            query = tf.cast(self.encoder_q(img_q, training=True), tf.float32)
            query = tf.math.l2_normalize(query, axis=1)

            l_pos = tf.einsum('nc,nc->n', query, tf.stop_gradient(key))[:,None]
            l_neg = tf.einsum('nc,ck->nk', query, self.queue)
            logits = tf.concat((l_pos, l_neg), axis=1)
            logits /= self.args.temperature

            loss_moco = self._loss(labels, logits, from_logits=True)
            loss_moco = tf.reduce_mean(loss_moco)

            loss_decay = sum(self.losses)

            loss = loss_moco + loss_decay
            total_loss = loss / self._num_workers

        trainable_vars = self.encoder_q.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(labels, logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss, 'loss_moco': loss_moco, 'weight_decay': loss_decay})
        self.update_queue(key)
        return results

    def concat_fn(self, strategy, key_per_replica):
        return tf.concat(key_per_replica.values, axis=0)

    def unshuffle_bn(self, key, unshuffle_idx):
        key_all_replica = self._replica_context.merge_call(self.concat_fn, args=(key,))
        unshuffle_idx_all_replica = self._replica_context.merge_call(self.concat_fn, args=(unshuffle_idx,))
        new_key_list = []
        for idx in unshuffle_idx_all_replica:
            new_key_list.append(tf.expand_dims(key_all_replica[idx], axis=0))
        key_orig = tf.concat(tuple(new_key_list), axis=0)
        key = key_orig[(self.args.batch_size//self._num_workers)*(self._replica_context.replica_id_in_sync_group):
                        (self.args.batch_size//self._num_workers)*(self._replica_context.replica_id_in_sync_group+1)]
        return key

    def reduce_key(self, key):
        key_all_replica = self._replica_context.merge_call(self.concat_fn, args=(key,))
        new_key_list = []
        for v in key_all_replica.values():
            new_key_list.append(tf.expand_dims(v, axis=0))
        all_key = tf.concat(tuple(new_key_list), axis=0)
        return all_key

    def update_queue(self, key):
        if self._num_workers > 1:
            key = self.reduce_key(key)
        self.queue = tf.concat([tf.transpose(key), self.queue], axis=-1)
        self.queue = self.queue[:,:self.args.num_negative]