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
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import utils
from estimate import estimate, read_image, eval_tensor


def main():
    matplotlib.rcParams.update({'font.size': args.fontsize})
    logdir = utils.get_logdir(config)
    symmetric_parts = utils.get_symmetric_parts(config)
    limbs_index = utils.get_limbs_index(config)
    height, width = config.getint('config', 'height'), config.getint('config', 'width')
    image = tf.placeholder(tf.float32, [1, height, width, 3], name='image')
    with open(logdir + '.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    limbs, parts = tf.import_graph_def(graph_def, input_map={'image:0': image}, return_elements=['limbs:0', 'parts:0'])
    with tf.Session() as sess:
        path = os.path.expanduser(os.path.expandvars(args.path))
        assert os.path.exists(path)
        if os.path.isfile(path):
            image_rgb, image_resized = read_image(path, height, width)
            _limbs, _parts = eval_tensor(sess, image, image_resized, [limbs, parts])
            estimate(config, image_rgb, symmetric_parts, limbs_index, _limbs, _parts)
            plt.show()
        else:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[-1].lower() in args.exts:
                        _path = os.path.join(dirpath, filename)
                        print(_path)
                        image_rgb, image_resized = read_image(_path, height, width)
                        _limbs, _parts = eval_tensor(sess, image, image_resized, [limbs, parts])
                        estimate(config, image_rgb, symmetric_parts, limbs_index, _limbs, _parts)
                        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='input image path')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-e', '--exts', nargs='+', default=['.jpg', '.png'])
    parser.add_argument('--level', default='info', help='logging level')
    parser.add_argument('--fontsize', default=7, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
