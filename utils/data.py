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
import tensorflow as tf
from . import preprocess

__ops__ = tf.load_op_library('openpose_ops.so')


def decode_image_labels(paths, num_parts):
    with tf.name_scope(inspect.stack()[0][3]):
        with tf.name_scope('parse_example'):
            reader = tf.TFRecordReader()
            _, serialized = reader.read(tf.train.string_input_producer(paths))
            example = tf.parse_single_example(serialized, features={
                'imageshape': tf.FixedLenFeature([3], tf.int64),
                'imagepath': tf.FixedLenFeature([], tf.string),
                'maskpath': tf.FixedLenFeature([], tf.string),
                'keypoints': tf.FixedLenFeature([], tf.string),
            })
        imageshape = example['imageshape']
        imageshape = tf.cast(imageshape, tf.int32, name='imageshape')
        with tf.name_scope('decode_image') as name:
            file = tf.read_file(example['imagepath'])
            image = tf.image.decode_jpeg(file, channels=3, name=name)
        with tf.name_scope('decode_mask') as name:
            file = tf.read_file(example['maskpath'])
            mask = tf.image.decode_jpeg(file, channels=1, name=name)
        with tf.name_scope('keypoints') as name:
            keypoints = tf.decode_raw(example['keypoints'], tf.int32)
            keypoints = tf.reshape(keypoints, [-1, num_parts, 3])
            keypoints = tf.cast(keypoints, tf.float32, name=name)
    return imageshape, image, mask, keypoints


def data_augmentation(config, image, mask, keypoints, width, height):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        scale = list(map(float, config.get(section, 'scale').split()))
        rotate = config.getfloat(section, 'rotate')
        image, mask, keypoints = __ops__.augmentation(image, mask, keypoints, [height, width], scale, rotate)
    return image, mask, keypoints


def data_augmentation_fixed(config, image, mask, keypoints, width, height):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        if config.getboolean(section, 'random_flip_horizontally'):
            image, mask, keypoints = preprocess.random_flip_horizontally(image, mask, keypoints, width)
        if config.getboolean(section, 'random_brightness'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_brightness(image, max_delta=63),
                lambda: image
            )
        if config.getboolean(section, 'random_saturation'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_saturation(image, lower=0.5, upper=1.5),
                lambda: image
            )
        if config.getboolean(section, 'random_hue'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_hue(image, max_delta=0.032),
                lambda: image
            )
        if config.getboolean(section, 'random_contrast'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_contrast(image, lower=0.5, upper=1.5),
                lambda: image
            )
        if config.getboolean(section, 'noise'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: image + tf.truncated_normal(tf.shape(image)) * tf.random_uniform([], 5, 15),
                lambda: image
            )
        grayscale_probability = config.getfloat(section, 'grayscale_probability')
        if grayscale_probability > 0:
            image = preprocess.random_grayscale(image, grayscale_probability)
    return image, mask, keypoints


def load_image_labels(config, paths, width, height, skeleton, num_parts):
    with tf.name_scope('batch'):
        _, image, mask, keypoints = decode_image_labels(paths, num_parts)
        image, mask, keypoints = data_augmentation(config, image, mask, keypoints, width, height)
        image = tf.cast(image, tf.float32)
        if config.getboolean('data_augmentation_fixed', 'enable'):
            image, mask, keypoints = data_augmentation_fixed(config, image, mask, keypoints, width, height)
        image = tf.clip_by_value(image, 0, 255)
        mask = tf.to_float(mask > 0)
    return image, (mask,)