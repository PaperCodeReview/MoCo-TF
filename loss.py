import tensorflow as tf

def crossentropy(args):
    def _loss(y_true, y_pred):
        if args.classes == 1:
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        else:
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return _loss