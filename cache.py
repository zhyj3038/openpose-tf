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
import importlib
import tensorflow as tf
import utils


def main():
    cachedir = utils.get_cachedir(config)
    os.makedirs(cachedir, exist_ok=True)
    mappers, _ = utils.get_dataset_mappers(config)
    for phase in args.phase:
        path = os.path.join(cachedir, phase) + '.tfrecord'
        tf.logging.info('write tfrecords file: ' + path)
        with tf.python_io.TFRecordWriter(path) as writer:
            for dataset in mappers:
                tf.logging.info('load %s data' % dataset)
                module = importlib.import_module('cache.' + dataset)
                func = getattr(module, 'cache')
                func(path, writer, mappers[dataset], args, config)
    tf.logging.info('%s data are saved into %s' % (str(args.phase), cachedir))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('-v', '--verify', action='store_true')
    parser.add_argument('-d', '--dump', action='store_true')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    with tf.Session() as sess:
        main()
