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
    def __init__(self, backbone, dim=128, K=65536, m=.999, T=.07, mlp=False):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        base_encoder = model_dict[backbone](include_top=False, weights=None)
        if mlp:
            x = Dense(512, name='fc1')(base_encoder.output)
            x = Activation('relu', name='fc1_relu')(x)
            x = Dense(dim, name='fc')(x)
        else:
            x = Dense(dim, name='fc')(base_encoder.output)
        
        self.encoder_q = tf.keras.models.Model(base_encoder.input, x, name='encoder')    
        self.encoder_k = tf.keras.models.clone_model(self.encoder_q)
        for i in range(len(self.encoder_q.layers)):
            self.encoder_k.get_layer(index=i).set_weights(
                self.encoder_q.get_layer(index=i).get_weights()
            )
        self.encoder_k.trainable = False

    def _momentum_update_key_encoder(self):
        for i in range(len(self.encoder_q.layers)):
            param_q = self.encoder_q.get_layer(index=i).get_weights()
            param_k = self.encoder_k.get_layer(index=i).get_weights()
            self.encoder_k.get_layer(index=i).set_weights(
                [param_k[j] * m + param_q[j] * (1. - m) for j in range(len(param_q))]
            )
            
    def call(self, inputs, training=False):
        im_q, im_k = inputs
        



def create_model(args, logger):

    backbone = model_dict[args.backbone](
        include_top=False,
        pooling='avg',
        weights=None,
        input_shape=(args.img_size, args.img_size, 3))

    def _softmax(x):
        return tf.math.exp(x/args.temperature) / tf.math.reduce_sum(tf.math.exp(x/args.temperature))

    x = Dense(args.classes)(backbone.output)
    x = Lambda(_softmax, name='main_output')(x)
        
    q_encoder = Model(backbone.input, x, name=args.backbone)
    k_encoder = tf.keras.models.clone_model(q_encoder)

    if args.snapshot:
        model.load_weights(args.snapshot)
        logger.info('Load weights at {}'.format(args.snapshot))
        
    return model

def momentum_update_model(args, encoder, m_encoder):
    pass