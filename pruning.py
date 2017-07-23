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
import shutil
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
import humanize
import utils


class Pruner(dict):
    def __init__(self, sess):
        self.sess = sess
    
    def __call__(self, value, var):
        name = var.op.name
        scope = os.path.dirname(name)
        if scope in self:
            indices = self[scope]
            tf.logging.info('use a cached indices in scope ' + scope)
        else:
            _value = np.abs(value)
            if len(value.shape) == 4:
                stat = np.mean(_value, axis=(0, 1, 2))
            elif len(value.shape) == 1:
                stat = _value
            indices = np.argsort(stat)[::-1][:var.get_shape()[-1].value]
            self[scope] = indices
        value = np.take(value, indices, -1)
        self.sess.run(var.assign(value))


def main():
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    height, width = config.getint('config', 'height'), config.getint('config', 'width')
    image = tf.placeholder(tf.float32, [1, height, width, 3], name='image')
    net = utils.parse_attr(config.get('backbone', 'dnn'))(config, image, train=True)
    stages = utils.parse_attr(config.get('stages', 'dnn'))(config, len(limbs_index), num_parts)
    limbs, parts = stages(net)
    tf.logging.warn(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
    global_step = tf.contrib.framework.get_or_create_global_step()
    variables = slim.get_variables_to_restore()
    with tf.Session() as sess:
        logdir = os.path.expanduser(os.path.expandvars(args.logdir))
        tf.logging.info('locating checkpoint in ' + logdir)
        checkpoint_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + checkpoint_path)
        pruner = Pruner(sess)
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        for var in variables:
            name = var.op.name
            value = reader.get_tensor(name)
            if np.array_equal(var.get_shape().as_list(), value.shape):
                sess.run(var.assign(value))
            else:
                tf.logging.info('pruning %s from %s into %s' % (name, str(value.shape), str(var.get_shape().as_list())))
                pruner(value, var)
        tf.logging.info('global_step=%d' % sess.run(global_step))
        saver = tf.train.Saver()
        logdir += args.suffix
        shutil.rmtree(logdir, ignore_errors=True)
        os.makedirs(logdir, exist_ok=True)
        model_path = os.path.join(logdir, 'model.ckpt')
        tf.logging.info('save model into ' + model_path)
        saver.save(sess, model_path)
        if args.summary:
            path = os.path.join(logdir, args.logname)
            summary_writer = tf.summary.FileWriter(path)
            summary_writer.add_graph(sess.graph)
            tf.logging.info('tensorboard --logdir ' + logdir)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', help='the soruce logdir')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('--suffix', default='_pruned', help='the suffix to be append in the destination logdir')
    parser.add_argument('-s', '--summary', action='store_true')
    parser.add_argument('--logname', default=time.strftime('%Y-%m-%d_%H-%M-%S'), help='the name of TensorBoard log')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
