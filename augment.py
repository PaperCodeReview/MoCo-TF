import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


mean_std = [[0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]]

class Augment:
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        self.mean, self.std = mean_std

    @tf.function
    def _augmentv1(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        x = self._crop(x, shape, coord)
        x = self._resize(x)
        x = self._random_grayscale(x, p=.2)
        x = self._color_jitter(x)
        x = self._random_hflip(x)
        x = self._standardize(x)
        return x

    @tf.function
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
        x -= self.mean
        x /= self.std
        return x

    def _crop(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_size=shape,
            bounding_boxes=coord,
            area_range=(.2, 1.),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        offset_height, offset_width, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        x = tf.slice(x, [offset_height, offset_width, 0], [target_height, target_width, 3])
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
        ''' Brightness in torchvision is implemented about multiplying the factor to image, 
            but tensorflow.image is just implemented about adding the factor to image.

        In tensorflow.image.adjust_brightness,
            For regular images, `delta` should be in the range `[0,1)`, 
            as it is added to the image in floating point representation, 
            where pixel values are in the `[0,1)` range.

        adjusted = math_ops.add(
            flt_image, math_ops.cast(delta, flt_image.dtype), name=name)

        However in torchvision docs,
        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.

        In torchvision.transforms.functional_tensor,
            return _blend(img, torch.zeros_like(img), brightness_factor)
            where _blend 
                return brightness * img1
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
            sig = np.random.uniform(sigma[0], sigma[1])
            x = tfa.image.gaussian_filter2d(x, sigma=sig)
        return x
