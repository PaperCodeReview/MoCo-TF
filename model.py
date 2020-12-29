import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


model_dict = {'resnet50': tf.keras.applications.ResNet50,}


def create_model(
    logger,
    backbone, 
    img_size,
    weight_decay=0.,
    use_bias=True,
    lincls=False,
    classes=1000,
    snapshot=None,
    freeze=False):

    base_encoder = model_dict[backbone](
        include_top=False,
        pooling='avg',
        weights=None,
        input_shape=(img_size, img_size, 3))
    
    if not use_bias:
        logger.info('\tConvert use_bias to False')
    if weight_decay > 0:
        logger.info(f'\tSet weight decay {weight_decay}')

    for layer in base_encoder.layers:
        if not use_bias:
            # exclude bias
            if hasattr(layer, 'use_bias'):
                layer.use_bias = False
                layer.bias = None
        
        if weight_decay > 0:
            # add l2 weight decay
            if hasattr(layer, 'kernel_regularizer'):
                setattr(layer, 'kernel_regularizer', l2(weight_decay))

    if weight_decay > 0:
        model_json = base_encoder.to_json()
        base_encoder = tf.keras.models.model_from_json(model_json)

    if lincls:
        base_encoder.load_weights(snapshot, by_name=True)
        if freeze:
            logger.info('Freeze the model!')
            for layer in base_encoder.layers:
                layer.trainable=False

        x = Dense(classes, use_bias=False)(base_encoder.output)
        model = Model(base_encoder.input, x, name=backbone)
        return model

    return base_encoder


class MoCo(tf.keras.Model):
    def __init__(
        self, 
        logger,
        backbone='resnet50',
        img_size=224,
        weight_decay=0.,
        use_bias=False,
        dim=128,
        K=65536,
        mlp=False,
        snapshot=None,
        *args,
        **kwargs):

        super(MoCo, self).__init__(*args, **kwargs)

        def _get_architecture(name=None):
            base_encoder = create_model(logger, backbone, img_size, weight_decay, use_bias)
            x = base_encoder.output
            x = Dense(dim, kernel_regularizer=l2(weight_decay))(x)
            if mlp:
                x = Activation('relu')(x)
                x = Dense(dim, kernel_regularizer=l2(weight_decay))(x)
            arch = Model(base_encoder.input, x, name=name)
            return arch
        
        logger.info('Set query encoder')
        self.encoder_q = _get_architecture('encoder_q_{}'.format(backbone))
        logger.info('Set key encoder')
        self.encoder_k = _get_architecture('encoder_k_{}'.format(backbone))

        if snapshot:
            self.encoder_q.load_weights(snapshot)
            logger.info('Load query weights at {}'.format(snapshot))
            self.encoder_k.load_weights(snapshot.replace('/query/', '/key/'))
            logger.info('Load key weights at {}'.format(snapshot.replace('/query/', '/key/')))
            self.queue = tf.constant(
                np.load(snapshot.replace('/query/', '/queue/').replace('.h5', '.npy')), dtype=tf.float32)
            logger.info('Load queue at {}'.format(snapshot.replace('/query/', '/queue/').replace('.h5', '.npy')))
        else:
            for i in range(len(self.encoder_q.layers)):
                self.encoder_k.get_layer(index=i).set_weights(
                    self.encoder_q.get_layer(index=i).get_weights())

            self.queue = tf.random.normal(shape=[dim, K])
            self.queue = tf.math.l2_normalize(self.queue, axis=0)

        self.encoder_k.trainable = False

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
        labels = tf.one_hot(tf.zeros(N, dtype=tf.int32), self.num_negative+1) # labels : (256, 65536+1)

        key = self.encoder_k(img_k, training=False)
        key = tf.math.l2_normalize(key, axis=1)
        if self.shuffle_bn and self.num_workers > 1:
            key = self.unshuffle_bn(key, unshuffle_idx)

        with tf.GradientTape() as tape:
            query = self.encoder_q(img_q, training=True)
            query = tf.math.l2_normalize(query, axis=1)
            l_pos = tf.reshape(
                tf.matmul(
                    tf.reshape(query, [N, 1, C]), 
                    tf.reshape(tf.stop_gradient(key), [N, C, 1])), 
                [N, 1])                                             # l_pos : (256, 1)
            l_neg = tf.matmul(query, self.queue)                    # l_neg : (256, 65536)
            logits = tf.concat((l_pos, l_neg), axis=1)              # logits : (256, 65536+1)
            logits /= self.temperature
            loss_moco = self.loss(labels, logits)
            loss_moco = tf.nn.compute_average_loss(
                loss_moco, global_batch_size=self.batch_size)
            loss_decay = tf.nn.scale_regularization_loss(sum(self.encoder_q.losses))
            loss = loss_moco + loss_decay

        trainable_vars = self.encoder_q.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(labels, logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss, 'loss_moco': loss_moco, 'weight_decay': loss_decay})

        if self.num_workers > 1:
            results.update({'key': self.reduce_key(key)})
        else:
            results.update({'key': key})
        
        return results

    def concat_fn(self, strategy, key_per_replica):
        return tf.concat(key_per_replica.values, axis=0)

    def unshuffle_bn(self, key, unshuffle_idx):
        replica_context = tf.distribute.get_replica_context()
        key_all_replica = replica_context.merge_call(self.concat_fn, args=(key,))
        unshuffle_idx_all_replica = replica_context.merge_call(self.concat_fn, args=(unshuffle_idx,))
        new_key_list = []
        for idx in unshuffle_idx_all_replica:
            new_key_list.append(tf.expand_dims(key_all_replica[idx], axis=0))
        key_orig = tf.concat(tuple(new_key_list), axis=0)
        key = key_orig[(self.batch_size//self.num_workers)*(replica_context.replica_id_in_sync_group):
                        (self.batch_size//self.num_workers)*(replica_context.replica_id_in_sync_group+1)]
        return key

    def reduce_key(self, key):
        replica_context = tf.distribute.get_replica_context()
        key_all_replica = replica_context.merge_call(self.concat_fn, args=(key,))
        new_key_list = []
        for v in key_all_replica.values():
            new_key_list.append(tf.expand_dims(v, axis=0))
        all_key = tf.concat(tuple(new_key_list), axis=0)
        return all_key