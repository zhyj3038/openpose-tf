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
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import utils


def main():
    skeleton = utils.get_skeleton(config)
    cachedir = utils.get_cachedir(config)
    with open(cachedir + '.parts', 'r') as f:
        num_parts = int(f.read())
    paths = [os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile]
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
    tf.logging.info('num_examples=%d' % num_examples)
    for path in paths:
        tf.logging.info('read ' + path)
        for serialized in tf.python_io.tf_record_iterator(path):
            example = tf.train.Example()
            example.ParseFromString(serialized)
            imagepath = example.features.feature['imagepath'].bytes_list.value[0].decode()
            image = scipy.misc.imread(imagepath)
            keypoints = np.fromstring(example.features.feature['keypoints'].bytes_list.value[0], dtype=np.int32).reshape([-1, num_parts, 3])
            maskpath = example.features.feature['maskpath'].bytes_list.value[0].decode()
            mask = scipy.misc.imread(maskpath)
            plt.imshow(image)
            plt.imshow(mask, alpha=args.alpha)
            for _keypoints in keypoints:
                for i, (x, y, v) in enumerate(_keypoints):
                    assert v >= 0
                    if v > 0:
                        plt.text(x, y, str(i), bbox=dict(facecolor=args.colors[v - 1], alpha=args.alpha))
                for i1, i2 in skeleton:
                    x1, y1, v1 = _keypoints[i1].T
                    x2, y2, v2 = _keypoints[i2].T
                    if v1 > 0 and v2 > 0:
                        plt.plot([x1, x2], [y1, y2])
            plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'])
    parser.add_argument('--colors', nargs='+', default=['r', 'w'])
    parser.add_argument('--level', default='info', help='logging level')
    parser.add_argument('--alpha', default=0.5)
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
