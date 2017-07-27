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
import csv
import operator
import numpy as np
import tensorflow as tf
import h5py
import utils


def check_mappers(mappers):
    for i in range(2):
        col = list(map(operator.itemgetter(i), mappers))
        if len(set(col)) != len(col):
            return False
    return True


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
            stages(net)
    if args.delete:
        tf.logging.warn('delete logging directory: ' + logdir)
        shutil.rmtree(logdir, ignore_errors=True)
    os.makedirs(logdir, exist_ok=True)
    with tf.Session() as sess:
        if args.logname:
            path = os.path.join(logdir, args.logname)
            summary_writer = tf.summary.FileWriter(path)
            summary_writer.add_graph(sess.graph)
            tf.logging.info('tensorboard --logdir ' + logdir)
        path = os.path.expanduser(os.path.expandvars(args.path))
        if args.mapper:
            with open(os.path.expanduser(os.path.expandvars(args.mapper)), 'r') as f:
                mappers = list(csv.reader(f, delimiter='\t'))
            assert check_mappers(mappers)
            with h5py.File(path, 'r') as f:
                for mapper in mappers:
                    if len(mapper) == 2:
                        name, key = mapper
                        transpose = None
                    else:
                        name, key, transpose = mapper
                        transpose = transpose.strip()
                        if transpose:
                            transpose = list(map(int, transpose.split(',')))
                        else:
                            transpose = None
                    tf.logging.info(key + '->' + name)
                    var = tf.get_default_graph().get_tensor_by_name(name + ':0')
                    val = f[key]
                    if transpose is None:
                        tf.logging.info(str(val.shape) + '->' + str(var.get_shape().as_list()))
                        val = np.array(val)
                    else:
                        shape = val.shape
                        try:
                            val = np.transpose(val, transpose)
                        except ValueError:
                            tf.logging.error('shape %s cannot be transposed via %s' % (str(val.shape), str(transpose)))
                            raise
                        tf.logging.info(str(shape) + '->' + str(val.shape) + '->' + str(var.get_shape().as_list()))
                    sess.run(tf.assign(var, val))
            dst = os.path.join(logdir, 'model.ckpt')
            tf.logging.info('save model into ' + dst)
            saver = tf.train.Saver()
            saver.save(sess, dst)
        else:
            for var in tf.global_variables():
                print(var.op.name + '\t' + ','.join(map(str, var.get_shape().as_list())))
            """
            with h5py.File(path, 'r') as f:
                for _dset in f.keys():
                    dset = f[_dset]
                    for _attr in dset:
                        attr = dset[_attr]
                        print('%s/%s\t' % (_dset, _attr) + ','.join(map(str, attr.shape)))
            """


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-m', '--mapper')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')
    parser.add_argument('--logname', default=time.strftime('%Y-%m-%d_%H-%M-%S'), help='the name of TensorBoard log')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
