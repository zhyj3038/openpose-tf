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
import numpy as np
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import humanize
import utils


def receptive_field_yx_min(changed):
    for i, row in enumerate(changed):
        for j, v in enumerate(row):
            if v > 0.5:
                return i, j


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
            limbs, parts = stages(net)
    tf.logging.info(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
    output = tf.concat([limbs, parts], -1)
    _, feature_height, feature_width, _ = output.get_shape().as_list()
    path = os.path.join(logdir, 'receptive_field') + args.ext
    tf.logging.info('load receptive file ' + path)
    changed = scipy.misc.imread(path)
    grid_height, grid_width = height / feature_height, width / feature_width
    plots = []
    def onclick(event):
        for p in plots:
            p.remove()
        ix = int(event.xdata / grid_width)
        iy = int(event.ydata / grid_height)
        plots.append(ax.add_patch(patches.Rectangle((ix * grid_width, iy * grid_height), grid_width, grid_height, linewidth=0, facecolor='black', alpha=.2)))
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image)
    ax.set_xticks(np.arange(0, width, grid_width))
    ax.set_yticks(np.arange(0, height, grid_height))
    ax.grid(which='both')
    ax.tick_params(labelbottom='off', labelleft='off')
    fig.canvas.mpl_connect('button_press_event', onclick)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='input image path')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('--level', default='info', help='logging level')
    parser.add_argument('-e', '--ext', default='.png')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
