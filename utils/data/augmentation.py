"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import inspect
import numpy as np
import tensorflow as tf


def flip_horizontally(image, mask, keypoints, width):
    scope = inspect.stack()[0][3]
    with tf.name_scope(scope):
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        x = keypoints[:, :, 0:1]
        remaining = keypoints[:, :, 1:]
        keypoints = tf.concat([width - x, remaining], 2)
    return image, mask, keypoints


def random_flip_horizontally(image, mask, keypoints, width, probability=0.5):
    scope = inspect.stack()[0][3]
    with tf.name_scope(scope):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: flip_horizontally(image, mask, keypoints, width)
        fn2 = lambda: (image, mask, keypoints)
        return tf.cond(pred, fn1, fn2)


def random_grayscale(image, probability=0.5):
    if probability <= 0:
        return image
    scope = inspect.stack()[0][3]
    with tf.name_scope(scope):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1] * (len(image.get_shape()) - 1) + [3])
        fn2 = lambda: image
        return tf.cond(pred, fn1, fn2)


def _random_brightness(config, image):
    name = inspect.stack()[0][3][1:]
    max_delta = config.getfloat('data_augmentation', name)
    image = eval('tf.image.' + name)(image, max_delta)
    return lambda: image


def _random_saturation(config, image):
    name = inspect.stack()[0][3][1:]
    lower, upper = list(map(float, config.get('data_augmentation', name).split()))
    image = eval('tf.image.' + name)(image, lower, upper)
    return lambda: image


def _random_hue(config, image):
    name = inspect.stack()[0][3][1:]
    max_delta = config.getfloat('data_augmentation', name)
    image = eval('tf.image.' + name)(image, max_delta)
    return lambda: image


def _random_contrast(config, image):
    name = inspect.stack()[0][3][1:]
    lower, upper = list(map(float, config.get('data_augmentation', name).split()))
    image = eval('tf.image.' + name)(image, lower, upper)
    return lambda: image


def _noise(config, image):
    name = inspect.stack()[0][3][1:]
    lower, upper = list(map(float, config.get('data_augmentation', name).split()))
    image = image + tf.truncated_normal(tf.shape(image)) * tf.random_uniform([], lower, upper)
    return lambda: image


def _random_grayscale(config, image):
    name = inspect.stack()[0][3][1:]
    probability = config.getfloat('data_augmentation', name)
    image = random_grayscale(image, probability)
    return lambda: image
