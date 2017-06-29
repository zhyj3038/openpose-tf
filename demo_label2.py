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
import matplotlib.pyplot as plt
import tensorflow as tf
import pyopenpose
import utils.data


def main():
    cachedir = utils.get_cachedir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    size_image = config.getint('config', 'height'), config.getint('config', 'width')
    size_label = utils.calc_backbone_size(config, size_image)
    tf.logging.info('size_image=%s, size_label=%s' % (str(size_image), str(size_label)))
    paths = [os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile]
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
    tf.logging.warn('num_examples=%d' % num_examples)
    threshold = config.getfloat('nms', 'threshold')
    limits = config.getint('nms', 'limits')
    with tf.Session() as sess:
        data = utils.data.load_data(config, paths, size_image, size_label, num_parts, limbs_index)
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        while True:
            image, _, keypoints, limbs, parts = sess.run(data)
            assert image.shape[:2] == size_image
            assert limbs.shape[:2] == size_label
            assert parts.shape[:2] == size_label
            image = image.astype(np.uint8)
            assert limbs.shape[2] == len(limbs_index) * 2
            assert parts.shape[2] == num_parts + 1
            scale_y, scale_x = size_image[0] / size_label[0], size_image[1] / size_label[1]
            for i, (i1, i2) in enumerate(limbs_index):
                fig, axes = plt.subplots(2, 2)
                for ax in axes[0]:
                    ax.imshow(image)
                part1, part2 = parts[:, :, i1], parts[:, :, i2]
                axes[0, 0].imshow(scipy.misc.imresize(part1, image.shape[:2]), alpha=args.alpha)
                axes[0, 1].imshow(scipy.misc.imresize(part2, image.shape[:2]), alpha=args.alpha)
                _limbs = limbs[:, :, i * 2:(i + 1) * 2]
                vmin, vmax = np.min(_limbs), np.max(_limbs)
                limb1, limb2 = np.transpose(_limbs, [2, 0, 1])
                axes[1, 0].imshow(scipy.misc.imresize(limb1, image.shape[:2]), vmin=vmin, vmax=vmax, alpha=args.alpha)
                axes[1, 1].imshow(scipy.misc.imresize(limb2, image.shape[:2]), vmin=vmin, vmax=vmax, alpha=args.alpha)
                peaks1 = pyopenpose.feature_peaks(parts[:, :, i1], threshold, limits)
                peaks2 = pyopenpose.feature_peaks(parts[:, :, i2], threshold, limits)
                for ax in axes[1]:
                    """
                    for points in keypoints:
                        x, y, v = points[i1].T
                        if v > 0:
                            ax.plot(x, y, 'x', color=args.color1)
                        x, y, v = points[i2].T
                        if v > 0:
                            ax.plot(x, y, 'x', color=args.color2)"""
                    for y, x, _ in peaks1:
                        tf.logging.info('(%f, %f)' % (limb1[y, x], limb2[y, x]))
                        ax.plot(x * scale_x, y * scale_y, '.', color=args.color1)
                    for y, x, _ in peaks2:
                        tf.logging.info('(%f, %f)' % (limb1[y, x], limb2[y, x]))
                        ax.plot(x * scale_x, y * scale_y, '.', color=args.color2)
                for ax in axes.flat:
                    ax.set_xlim([0, size_image[1] - 1])
                    ax.set_ylim([size_image[0] - 1, 0])
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.canvas.set_window_title('limb%d (%d-%d)' % (i, i1, i2))
                plt.show()
        coord.request_stop()
        coord.join(threads)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'])
    parser.add_argument('--color1', default='r')
    parser.add_argument('--color2', default='g')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
