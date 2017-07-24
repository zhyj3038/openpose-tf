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

import os
import argparse
import configparser
import operator
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
import matplotlib.pyplot as plt
import humanize
import utils.preprocess
import utils.visualize
import pyopenpose


def estimate(config, image, symmetric_parts, limbs_index, limbs, parts):
    threshold = config.getfloat('nms', 'threshold')
    radius_scale = config.getfloat('cluster', 'radius_scale')
    steps = config.getint('integration', 'steps')
    min_score = config.getfloat('integration', 'min_score')
    min_count = config.getint('integration', 'min_count')
    cluster_min_score = config.getfloat('cluster', 'min_score')
    cluster_min_count = config.getint('cluster', 'min_count')
    clusters = pyopenpose.estimate(radius_scale, symmetric_parts, limbs_index, limbs, parts, threshold, steps, min_score, min_count, cluster_min_score, cluster_min_count)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image)
    scale_y, scale_x = utils.preprocess.calc_image_scale(parts.shape[:2], image.shape[:2])
    utils.visualize.draw_estimation(ax, scale_y, scale_x, clusters)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def read_image(path, height, width):
    image = utils.preprocess.read_image(path)
    image = np.array(np.uint8(image))
    if len(image.shape) == 2:
        image = np.repeat(np.expand_dims(image, -1), 3, 2)
    return image, utils.preprocess.resize(image, height, width)


def eval_tensor(sess, image, _image, tensors):
    _image = _image.astype(np.float32)
    _image = utils.preprocess.per_image_standardization(_image)
    feed_dict = {image: np.expand_dims(_image, 0)}
    _tensors = sess.run(tensors, feed_dict)
    return tuple(map(operator.itemgetter(0), _tensors))


def main():
    matplotlib.rcParams.update({'font.size': args.fontsize})
    logdir = utils.get_logdir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    symmetric_parts = utils.get_symmetric_parts(config)
    limbs_index = utils.get_limbs_index(config)
    height, width = config.getint('config', 'height'), config.getint('config', 'width')
    image = tf.placeholder(tf.float32, [1, height, width, 3], name='image')
    net = utils.parse_attr(config.get('backbone', 'dnn'))(config, image, train=True)
    with tf.variable_scope('stages'):
        stages = utils.parse_attr(config.get('stages', 'dnn'))(config, len(limbs_index), num_parts)
        try:
            count = config.getint('stages', 'count')
        except configparser.NoOptionError:
            count = stages.count
        for _ in range(count):
            limbs, parts = stages(net)
    limbs = tf.check_numerics(limbs, limbs.op.name)
    parts = tf.check_numerics(parts[:, :, :, :-1], parts.op.name) # drop background channel
    tf.logging.info(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
    with tf.Session() as sess:
        tf.logging.info('locating checkpoint in ' + logdir)
        checkpoint_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + checkpoint_path)
        slim.assign_from_checkpoint_fn(checkpoint_path, tf.global_variables())(sess)
        path = os.path.expanduser(os.path.expandvars(args.path))
        if os.path.isfile(path):
            image_rgb, image_resized = read_image(path, height, width)
            _limbs, _parts = eval_tensor(sess, image, image_resized, [limbs, parts])
            estimate(config, image_rgb, symmetric_parts, limbs_index, _limbs, _parts)
            plt.show()
        else:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[-1].lower() in args.exts:
                        _path = os.path.join(dirpath, filename)
                        print(_path)
                        image_rgb, image_resized = read_image(_path, height, width)
                        _limbs, _parts = eval_tensor(sess, image, image_resized, [limbs, parts])
                        estimate(config, image_rgb, symmetric_parts, limbs_index, _limbs, _parts)
                        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='input image path')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-e', '--exts', nargs='+', default=['.jpg', '.png'])
    parser.add_argument('--level', default='info', help='logging level')
    parser.add_argument('--fontsize', default=7, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
