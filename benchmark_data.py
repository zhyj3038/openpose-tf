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
import tensorflow as tf
import tqdm
import utils.data


def gen(sess, data):
    while True:
        yield sess.run(data)


def main():
    cachedir = utils.get_cachedir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    height, width = config.getint('config', 'height'), config.getint('config', 'width')
    feature_height, feature_width = utils.calc_downsampling_size(config.get('backbone', 'dnn'), height, width)
    paths = [os.path.join(cachedir, phase + '.tfrecord') for phase in args.phase]
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
    tf.logging.warn('num_examples=%d' % num_examples)
    with tf.Session() as sess:
        with tf.name_scope('batch'):
            image, mask, _, limbs, parts = utils.data.load_data(config, paths, height, width, feature_height, feature_width, num_parts, limbs_index)
            batch = tf.train.shuffle_batch([image, mask, limbs, parts], batch_size=args.batch_size,
                capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count()
            )
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for _ in tqdm.tqdm(gen(sess, batch)):
            pass
        coord.request_stop()
        coord.join(threads)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val'])
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
