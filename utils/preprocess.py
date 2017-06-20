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
import cv2


def per_image_standardization(image):
    stddev = np.std(image)
    return (image - np.mean(image)) / max(stddev, 1.0 / np.sqrt(np.multiply.reduce(image.shape)))


def resize(image, size):
    height, width, _ = image.shape
    _width, _height = size
    if height / width > _height / _width:
        scale = _height / height
    else:
        scale = _width / width
    m = np.eye(2, 3)
    m[0, 0] = scale
    m[1, 1] = scale
    return cv2.warpAffine(image, m, size, flags=cv2.INTER_CUBIC)


def flip_horizontally(image, mask, keypoints, width):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        x = keypoints[:, :, 0:1]
        remaining = keypoints[:, :, 1:]
        keypoints = tf.concat([width - x, remaining], 2)
    return image, mask, keypoints


def random_flip_horizontally(image, mask, keypoints, width, probability=0.5):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: flip_horizontally(image, mask, keypoints, width)
        fn2 = lambda: (image, mask, keypoints)
        return tf.cond(pred, fn1, fn2)


def random_grayscale(image, probability=0.5):
    if probability <= 0:
        return image
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1] * (len(image.get_shape()) - 1) + [3])
        fn2 = lambda: image
        return tf.cond(pred, fn1, fn2)
