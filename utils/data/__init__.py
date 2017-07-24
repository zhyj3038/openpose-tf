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
import configparser
import tensorflow as tf
from . import augmentation

__ops__ = tf.load_op_library('openpose_ops.so')


def decode_image_label(config, paths, num_parts):
    with tf.name_scope(inspect.stack()[0][3]):
        with tf.name_scope('parse_example'):
            reader = tf.TFRecordReader()
            _, serialized = reader.read(tf.train.string_input_producer(paths))
            example = tf.parse_single_example(serialized, features={
                'imagepath': tf.FixedLenFeature([], tf.string),
                'maskpath': tf.FixedLenFeature([], tf.string),
                'keypoints': tf.FixedLenFeature([], tf.string),
            })
        with tf.name_scope('decode_image') as name:
            file = tf.read_file(example['imagepath'])
            image = tf.image.decode_image(file, config.getint('config', 'channels'), name=name)
        with tf.name_scope('decode_mask') as name:
            file = tf.read_file(example['maskpath'])
            mask = tf.image.decode_jpeg(file, channels=1, name=name)
        with tf.name_scope('keypoints') as name:
            keypoints = tf.decode_raw(example['keypoints'], tf.int32)
            keypoints = tf.reshape(keypoints, [-1, num_parts, 3])
            keypoints = tf.cast(keypoints, tf.float32, name=name)
    return image, mask, keypoints


def data_augmentation(config, image, mask, keypoints, width):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        if config.getboolean(section, 'random_flip_horizontally'):
            image, mask, keypoints = augmentation.random_flip_horizontally(image, mask, keypoints, width)
        if config.getboolean(section, 'enable'):
            for name in config.get(section, 'sequence').split():
                try:
                    with tf.name_scope(name):
                        image = tf.cond(
                            tf.random_uniform([], name='enable_probability') < config.getfloat(section, 'enable_probability'),
                            eval('augmentation._' + name)(config, image),
                            lambda: image
                        )
                except configparser.NoOptionError:
                    tf.logging.warn(name + ' disabled')
    return image, mask, keypoints


def load_data(config, paths, height, width, feature_height, feature_width, num_parts, limbs_index):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        image, mask, keypoints = decode_image_label(config, paths, num_parts)
        scale = list(map(float, config.get('data_augmentation', 'scale').split()))
        rotate = config.getfloat('data_augmentation', 'rotate')
        fill = config.getint('data_augmentation', 'fill')
        image, mask, keypoints = __ops__.augmentation(image, mask, keypoints, [height, width], [feature_height, feature_width], scale, rotate, fill)
        shape = image.get_shape().as_list()
        assert shape[:-1] == [height, width]
        if shape[-1] is None:
            image = tf.reshape(image, shape[:2] + [config.getint('config', 'channels')], name='fix_channels')
        assert mask.get_shape().as_list()[:-1] == [feature_height, feature_width]
        image = tf.cast(image, tf.float32)
        image, mask, keypoints = data_augmentation(config, image, mask, keypoints, width)
        image = tf.clip_by_value(image, 0, 255)
        mask = tf.to_float(mask > 127)
    sigma_parts = config.getfloat('label', 'sigma_parts') * max(feature_height, feature_width)
    sigma_limbs = config.getfloat('label', 'sigma_limbs') * max(feature_height, feature_width)
    limbs, parts = __ops__.label([height, width], [feature_height, feature_width], keypoints, limbs_index, sigma_parts, sigma_limbs)
    assert limbs.get_shape().as_list()[:-1] == [feature_height, feature_width]
    assert parts.get_shape().as_list()[:-1] == [feature_height, feature_width]
    return image, mask, keypoints, limbs, parts
