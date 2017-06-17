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


def main():
    skeleton = utils.get_skeleton(config)
    cachedir = utils.get_cachedir(config)
    with open(cachedir + '.parts', 'r') as f:
        num_parts = int(f.read())
    width = config.getint('config', 'width')
    height = config.getint('config', 'height')
    tf.logging.info('(width, height)=(%d, %d)' % (width, height))
    paths = [os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile]
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
    tf.logging.warn('num_examples=%d' % num_examples)
    with tf.Session() as sess:
        with tf.name_scope('batch'):
            image_rgb, labels = utils.data.load_image_labels(config, paths, width, height, skeleton, num_parts)
            batch = tf.train.shuffle_batch((image_rgb,) + labels, batch_size=args.batch_size,
                capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count()
            )
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        while True:
            cnt = 0
            while True:
                batch_image, batch_labels = sess.run([batch[0], batch[1:]])
                print(cnt)
                cnt += 1
            batch_image = batch_image.astype(np.uint8)
            fig, axes = plt.subplots(args.batch_size, 4)
            for b, (_axes, image) in enumerate(zip(axes, batch_image)):
                _axes[0].imshow(image)
            for ax in axes.flat:
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
    parser.add_argument('-g', '--grid', action='store_true')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
