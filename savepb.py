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
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import humanize
import utils


def main():
    logdir = utils.get_logdir(config)
    _, num_parts = utils.get_dataset_mappers(config)
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
    limbs, parts = tf.identity(limbs, 'limbs'), tf.identity(parts[:, :, :, :-1], 'parts') # drop background channel
    tf.logging.info(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
    with tf.Session() as sess:
        tf.logging.info('locating checkpoint in ' + logdir)
        checkpoint_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + checkpoint_path)
        variables = slim.get_variables_to_restore()
        tf.logging.info('restore %d variables' % len(variables))
        slim.assign_from_checkpoint_fn(checkpoint_path, variables)(sess)
        graph = convert_variables_to_constants(sess, sess.graph_def, [limbs.op.name, parts.op.name])
        path = logdir + '.pb'
        tf.logging.info('freeze model into ' + path)
        tf.train.write_graph(graph, os.path.dirname(path), os.path.basename(path), as_text=False)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
