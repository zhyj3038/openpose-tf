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
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils.data
import utils.visualize


def main():
    limbs = utils.get_limbs(config)
    cachedir = utils.get_cachedir(config)
    with open(cachedir + '.parts', 'r') as f:
        num_parts = int(f.read())
    size_image = config.getint('config', 'height'), config.getint('config', 'width')
    size_label = (size_image[0] // 8, size_image[1] // 8)
    tf.logging.info('size_image=%s, size_label=%s' % (str(size_image), str(size_label)))
    batch_size = args.rows * args.cols
    paths = [os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile]
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
    tf.logging.warn('num_examples=%d' % num_examples)
    with tf.Session() as sess:
        with tf.name_scope('batch'):
            image, mask, _, label = utils.data.load_data(config, paths, size_image, size_label, num_parts, limbs)
            batch = tf.train.shuffle_batch([image, mask, label], batch_size=batch_size,
                capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count()
            )
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        while True:
            _batch = sess.run(batch)
            fig, axes = plt.subplots(args.rows, args.cols)
            for ax, image, mask, label in zip(*([axes.flat] + _batch)):
                assert image.shape[:2] == size_image
                assert label.shape[:2] == size_label
                image = image.astype(np.uint8)
                utils.visualize.draw_mask(image, mask)
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            plt.show()
        coord.request_stop()
        coord.join(threads)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'])
    parser.add_argument('--rows', default=5, type=int)
    parser.add_argument('--cols', default=5, type=int)
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
