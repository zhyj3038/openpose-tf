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

import argparse
import configparser
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import humanize
import utils


def main():
    analyze = [
        ('name', lambda sess, var, cache: var.op.name),
        ('_shape', lambda sess, var, cache: var.get_shape().as_list()),
        ('shape', lambda sess, var, cache: 'x'.join(map(str, cache['_shape']))),
        ('bytes', lambda sess, var, cache: np.multiply.reduce(cache['_shape']) * var.dtype.size),
        ('size', lambda sess, var, cache: humanize.naturalsize(cache['bytes'])),
    ]
    if not args.size_only:
        analyze += [
            ('_val', lambda sess, var, cache: sess.run(var)),
            ('_abs_val', lambda sess, var, cache: np.abs(cache['_val'])),
            ('abs_val_mean', lambda sess, var, cache: np.mean(cache['_abs_val'])),
            ('_abs_val_mean_lower', lambda sess, var, cache: np.mean(cache['_abs_val'], axis=(0, 1, 2)) if len(cache['_abs_val'].shape) == 4 else cache['_abs_val'] if len(cache['_abs_val'].shape) == 1 else None),
            ('abs_val_mean_lower', lambda sess, var, cache: np.min(cache['_abs_val_mean_lower']) if '_abs_val_mean_lower' in cache else None),
            ('_abs_val_min_lower', lambda sess, var, cache: np.min(cache['_abs_val'], axis=(0, 1, 2)) if len(cache['_abs_val'].shape) == 4 else cache['_abs_val'] if len(cache['_abs_val'].shape) == 1 else None),
            ('abs_val_min_lower', lambda sess, var, cache: np.min(cache['_abs_val_min_lower']) if '_abs_val_min_lower' in cache else None),
        ]
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
    tf.logging.warn(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
    if not args.nohead:
        print('\t'.join(['index'] + [name for name, _ in analyze if name[0] != '_']))
    with tf.Session() as sess:
        if not args.size_only:
            tf.logging.info('locating checkpoint in ' + logdir)
            checkpoint_path = tf.train.latest_checkpoint(logdir)
            tf.logging.info('load ' + checkpoint_path)
            slim.assign_from_checkpoint_fn(checkpoint_path, tf.global_variables())(sess)
        for index, var in enumerate(tf.global_variables()):
            row = [str(index)]
            cache = {}
            for name, func in analyze:
                try:
                    r = func(sess, var, cache)
                except:
                    tf.logging.error(cache)
                    raise
                if r is not None:
                    cache[name] = r
                if name[0] != '_':
                    row.append(r if isinstance(r, str) else str(r))
            print('\t'.join(row))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('--level', default='info', help='logging level')
    parser.add_argument('--nohead', action='store_true')
    parser.add_argument('-s', '--size_only', action='store_true', help='only analyze variable sizes')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
