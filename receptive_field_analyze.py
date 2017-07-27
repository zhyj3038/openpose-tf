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
import itertools
import operator
import numpy as np
import scipy.misc
import tensorflow as tf
import tqdm
import humanize
import utils


def fake_image(i, j, height, width, channels=3, fill=1):
    image = np.zeros([height, width, 3])
    image[i, j, :] = fill
    return image


def main():
    logdir = utils.get_logdir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    height, width = config.getint('config', 'height'), config.getint('config', 'width')
    image = tf.placeholder(tf.float32, [args.batch_size, height, width, 3], name='image')
    net = utils.parse_attr(config.get('backbone', 'dnn'))(config, image, train=True)
    with tf.variable_scope('stages'):
        stages = utils.parse_attr(config.get('stages', 'dnn'))(config, len(limbs_index), num_parts)
        try:
            count = config.getint('stages', 'count')
        except configparser.NoOptionError:
            count = stages.count
        for _ in range(count):
            limbs, parts = stages(net)
    if count > 0:
        net = tf.concat([limbs, parts], -1)
    else:
        tf.logging.warn('no stages')
    tf.logging.info(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
    _, feature_height, feature_width, _ = net.get_shape().as_list()
    output = net[:, feature_height // 2, feature_width // 2, :]
    changed = np.empty([height, width], dtype=np.bool)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        _output0 = sess.run(output, {image: np.zeros([args.batch_size, height, width, 3])})
        points = [(i, j) for i in range(height) for j in range(width)]
        batches = [list(map(operator.itemgetter(1), group)) for _, group in itertools.groupby(enumerate(points), lambda item: item[0] // args.batch_size)]
        for batch in tqdm.tqdm(batches):
            assert len(batch) <= args.batch_size
            if len(batch) < args.batch_size:
                batch += [batch[-1]] * (args.batch_size - len(batch))
            _image = [fake_image(i, j, height, width) for i, j in batch]
            _image = np.stack(_image, 0)
            _output = sess.run(output, {image: _image})
            _changed = [np.any(c) for c in _output != _output0]
            for (i, j), c in zip(batch, _changed):
                changed[i, j] = c
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, 'receptive_field') + args.ext
    scipy.misc.imsave(path, changed)
    tf.logging.info('receptive file saved into ' + path)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('--level', default='info', help='logging level')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('-e', '--ext', default='.png')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
