import random
import tensorflow as tf
import tensorflow_addons as tfa


mean_std = {
    'cub': [[0.48552202, 0.49934904, 0.43224954], 
            [0.18172876, 0.18109447, 0.19272076]],
    'cifar100': [[0.50707516, 0.48654887, 0.44091784], 
                 [0.20079844, 0.19834627, 0.20219835]],
    'imagenet': [[0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225]]
}

class Augment:
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        self.mean, self.std = mean_std[args.dataset]

    def _augmentv1(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        x = self._crop(x, shape, coord)
        x = self._resize(x)
        x = self._random_grayscale(x, p=.2)
        x = self._color_jitter(x)
        x = self._random_hflip(x)
        x = self._standardize(x)
        return x

    def _augmentv2(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        x = self._crop(x, shape, coord)
        x = self._resize(x)
        x = self._random_color_jitter(x, p=.8)
        x = self._random_grayscale(x, p=.2)
        x = self._random_gaussian_blur(x, sigma=[.1, 2.], p=.5)
        x = self._random_hflip(x)
        x = self._standardize(x)
        return x

    def _standardize(self, x):
        x = tf.cast(x, tf.float32)
        x /= 255.
        if self.args.standardize == "minmax1":
            pass
        elif self.args.standardize == "minmax2":
            x -= 0.5
            x /= 0.5
        elif self.args.standardize == "norm":
            x -= self.mean
            x /= self.std
        elif self.args.standardize == "eachnorm":
            x = (x-tf.math.reduce_mean(x))/tf.math.reduce_std(x)
        else:
            raise ValueError()
        return x

    def _crop(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        begin, size, bboxes = tf.image.sample_distorted_bounding_box(
            image_size=shape,
            bounding_boxes=coord,
            area_range=(.2, 1.),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        x = tf.slice(x, begin, size)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _resize(self, x):
        x = tf.image.resize(x, (self.args.img_size, self.args.img_size))
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _color_jitter(self, x, _jitter_idx=[0, 1, 2, 3]):
        random.shuffle(_jitter_idx)
        _jitter_list = [
            self._brightness,
            self._contrast,
            self._saturation,
            self._hue]
        for idx in _jitter_idx:
            x = _jitter_list[idx](x)
        return x

    def _random_color_jitter(self, x, p=.8):
        if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)):
            x = self._color_jitter(x)
        return x

    def _brightness(self, x):
        ''' Brightness in torchvision is implemented about multiplying factor to image, 
            but tensorflow.image is just implemented about adding factor to image.
        '''
        # x = tf.image.random_brightness(x, max_delta=self.args.brightness)
        x = tf.cast(x, tf.float32)
        delta = tf.random.uniform(
            shape=[], 
            minval=1-self.args.brightness,
            maxval=1+self.args.brightness,
            dtype=tf.float32)

        x *= delta
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _contrast(self, x):
        x = tf.image.random_contrast(x, lower=max(0, 1-self.args.contrast), upper=1+self.args.contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _saturation(self, x):
        x = tf.image.random_saturation(x, lower=max(0, 1-self.args.contrast), upper=1+self.args.contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _hue(self, x):
        x = tf.image.random_hue(x, max_delta=self.args.hue)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _grayscale(self, x):
        return tf.image.rgb_to_grayscale(x) # after expand_dims

    def _random_grayscale(self, x, p=.2):
        if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)):
            x = self._grayscale(x)
            x = tf.tile(x, [1, 1, 3])
        return x

    def _random_hflip(self, x):
        return tf.image.random_flip_left_right(x)

    def _random_gaussian_blur(self, x, sigma=[.1, 2.], p=.5):
        if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)):
            sig = tf.random.uniform(shape=[], minval=sigma[0], maxval=sigma[1])
            x = tfa.image.gaussian_filter2d(x, sigma=sig)
        return x
