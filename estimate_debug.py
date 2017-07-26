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
import shutil
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
import matplotlib.pyplot as plt
import humanize
import utils.preprocess
import utils.visualize
from estimate import read_image, eval_tensor, estimate
import pyopenpose


def dump(path, image, symmetric_parts, limbs_index, limbs, parts):
    path = os.path.expanduser(os.path.expandvars(path))
    tf.logging.warn('dump feature map into ' + path)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'symmetric_parts.txt'), 'w') as f:
        for channels in symmetric_parts:
            f.write(' '.join(map(str, channels)) + '\n')
    np.savetxt(os.path.join(path, 'limbs_index.tsv'), limbs_index, fmt='%d', delimiter='\t')
    scipy.misc.imsave(os.path.join(path, 'image.jpg'), image)
    np.save(os.path.join(path, 'limbs.npy'), limbs)
    np.save(os.path.join(path, 'parts.npy'), parts)


def debug_nms(root, image, parts, threshold, alpha=0.5):
    os.makedirs(root, exist_ok=True)
    scale_y, scale_x = utils.preprocess.calc_image_scale(parts.shape[:2], image.shape[:2])
    maxsize = max(image.shape[:2])
    for index, feature in enumerate(np.transpose(parts, [2, 0, 1])):
        peaks = pyopenpose.feature_peaks(feature, threshold)
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(image)
        ax.imshow(scipy.misc.imresize(feature, (maxsize, maxsize)), alpha=alpha)
        utils.visualize.draw_peaks(ax, scale_y, scale_x, peaks, str(index))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('%d points' % len(peaks))
        fig.savefig(os.path.join(root, 'part%03d' % index) + args.ext)
        plt.close(fig)
    peaks = pyopenpose.featuremap_peaks(parts, threshold)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image)
    for index, _peaks in enumerate(peaks):
        utils.visualize.draw_peaks(ax, scale_y, scale_x, _peaks, str(index))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(os.path.join(root, 'parts') + args.ext)
    plt.close(fig)


def debug_connection(root, image, radius_scale, symmetric_parts, limbs_index, limbs, parts, threshold, steps, min_score, min_count, alpha=0.5, linewidth=10):
    os.makedirs(root, exist_ok=True)
    scale_y, scale_x = utils.preprocess.calc_image_scale(parts.shape[:2], image.shape[:2])
    peaks = pyopenpose.featuremap_peaks(parts, threshold)
    assert len(peaks) == parts.shape[-1]
    _limbs = np.reshape(limbs, limbs.shape[:2] + (-1, 2))
    _limbs = np.transpose(_limbs, [2, 3, 0, 1])
    maxsize = max(image.shape[:2])
    for index, (color, (i1, i2), (limb_x, limb_y)) in enumerate(zip(itertools.cycle(prop['color'] for prop in plt.rcParams['axes.prop_cycle']), limbs_index, _limbs)):
        figs = [plt.figure() for _ in range(3)]
        axes = [fig.gca() for fig in figs]
        for ax in axes:
            ax.imshow(image)
        axes[0].imshow(scipy.misc.imresize(limb_x, (maxsize, maxsize)), alpha=alpha)
        axes[1].imshow(scipy.misc.imresize(limb_y, (maxsize, maxsize)), alpha=alpha)
        for i, _peaks in enumerate(peaks):
            for ax in axes[:2]:
                for j, (y, x, _) in enumerate(_peaks):
                    y, x = y * scale_y, x * scale_x
                    ax.text(x, y, '%d_%d' % (i, j) if i in (i1, i2) else str(i), bbox=dict(facecolor='r' if i == i1 else 'g' if i == i2 else 'w', alpha=alpha), ha='center', va='center')
        peaks1 = peaks[i1]
        peaks2 = peaks[i2]
        connections = pyopenpose.calc_limb_score(index * 2, limbs, peaks1, peaks2, steps, min_score, min_count, radius_scale)
        for ax in axes[:2]:
            for p1, p2, score in connections:
                y1, x1, _ = peaks1[p1]
                y2, x2, _ = peaks2[p2]
                y1, x1 = y1 * scale_y, x1 * scale_x
                y2, x2 = y2 * scale_y, x2 * scale_x
                ax.plot([x1, x2], [y1, y2])
        if min_score > 0 and min_count > 0:
            pyopenpose.filter_connections(connections, peaks1, peaks2, peaks, symmetric_parts[i1], symmetric_parts[i2], radius_scale)
        max_score = max(connections, key=lambda item: item[-1])[-1] if connections else 0
        ax = axes[-1]
        for p1, p2, score in connections:
            y1, x1, _ = peaks1[p1]
            y2, x2, _ = peaks2[p2]
            y1, x1 = y1 * scale_y, x1 * scale_x
            y2, x2 = y2 * scale_y, x2 * scale_x
            ax.text(x1, y1, str(i1), bbox=dict(facecolor='r', alpha=alpha), ha='center', va='center')
            ax.text(x2, y2, str(i2), bbox=dict(facecolor='g', alpha=alpha), ha='center', va='center')
            if max_score > 0:
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth * score / max_score)
        ax.set_title('%d-%d' % (i1, i2))
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        path = os.path.join(root, 'limb%03d' % index)
        os.makedirs(path, exist_ok=True)
        for i, fig in enumerate(figs[:2]):
            fig.savefig(os.path.join(path, str(i)) + args.ext)
        figs[-1].savefig(path + args.ext)
        for fig in figs:
            plt.close(fig)


def debug_clusters(root, image, radius_scale, symmetric_parts, limbs_index, limbs, parts, threshold, steps, min_score, min_count, linewidth=10):
    scale_y, scale_x = utils.preprocess.calc_image_scale(parts.shape[:2], image.shape[:2])
    peaks = pyopenpose.featuremap_peaks(parts, threshold)
    assert len(peaks) == parts.shape[-1]
    clusters = pyopenpose.clustering(radius_scale, symmetric_parts, limbs_index, limbs, peaks, steps, min_score, min_count)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image)
    max_score = max(clusters, key=lambda item: item[1])[1] if clusters else 0
    for color, (points, score, count) in zip(itertools.cycle(prop['color'] for prop in plt.rcParams['axes.prop_cycle']), clusters):
        assert len(points) == len(peaks)
        assert sum(1 for p in points if p >= 0) == count
        assert count > 0
        _linewidth = linewidth * score / max_score
        for i1, i2 in limbs_index:
            p1, p2 = points[i1], points[i2]
            if p1 >= 0 and p2 >= 0:
                y1, x1, _ = peaks[i1][p1]
                y2, x2, _ = peaks[i2][p2]
                y1, x1 = y1 * scale_y, x1 * scale_x
                y2, x2 = y2 * scale_y, x2 * scale_x
                ax.plot([x1, x2], [y1, y2], '.', color=color)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=_linewidth)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(root + args.ext)
    plt.close(fig)


def debug(config, root, image, symmetric_parts, limbs_index, limbs, parts):
    threshold = config.getfloat('nms', 'threshold')
    radius_scale = config.getfloat('cluster', 'radius_scale')
    steps = config.getint('integration', 'steps')
    min_score = config.getfloat('integration', 'min_score')
    min_count = config.getint('integration', 'min_count')
    if args.nms:
        tf.logging.info('debug NMS')
        debug_nms(os.path.join(root, 'nms'), image, parts, threshold)
    if args.limbs:
        tf.logging.info('debug score')
        debug_connection(os.path.join(root, 'score'), image, radius_scale, symmetric_parts, limbs_index, limbs, parts, threshold, steps, 0, 1)
        tf.logging.info('debug connection')
        debug_connection(os.path.join(root, 'connection'), image, radius_scale, symmetric_parts, limbs_index, limbs, parts, threshold, steps, min_score, min_count)
    tf.logging.info('debug clusters')
    debug_clusters(os.path.join(root, 'clusters'), image, radius_scale, symmetric_parts, limbs_index, limbs, parts, threshold, steps, min_score, min_count)
    fig = estimate(config, image, symmetric_parts, limbs_index, limbs, parts)
    fig.savefig(os.path.join(root, 'estimate') + args.ext)
    plt.close(fig)


def main():
    matplotlib.rcParams.update({'font.size': args.fontsize})
    logdir = utils.get_logdir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    symmetric_parts = utils.get_symmetric_parts(config)
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
    limbs = tf.check_numerics(limbs, limbs.op.name)
    parts = tf.check_numerics(parts[:, :, :, :-1], parts.op.name) # drop background channel
    tf.logging.info(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
    with tf.Session() as sess:
        tf.logging.info('locating checkpoint in ' + logdir)
        checkpoint_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + checkpoint_path)
        variables = slim.get_variables_to_restore()
        slim.assign_from_checkpoint_fn(checkpoint_path, variables)(sess)
        path = os.path.expanduser(os.path.expandvars(args.path))
        assert os.path.isfile(path), path + ' is not a file'
        root = os.path.splitext(path)[0]
        if args.delete:
            tf.logging.warn('delete root directory: ' + root)
            shutil.rmtree(root, ignore_errors=True)
        image_rgb, image_resized = read_image(path, height, width)
        _limbs, _parts = eval_tensor(sess, image, image_resized, [limbs, parts])
        dump(root, image_resized, symmetric_parts, limbs_index, _limbs, _parts)
        debug(config, root, image_rgb, symmetric_parts, limbs_index, _limbs, _parts)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='input image path')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-d', '--delete', action='store_true', help='delete root')
    parser.add_argument('-e', '--ext', default='.svg')
    parser.add_argument('--nms', action='store_true')
    parser.add_argument('--limbs', action='store_true')
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
