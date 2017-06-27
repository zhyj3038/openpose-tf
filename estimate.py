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
import operator
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
import matplotlib.pyplot as plt
import utils.preprocess
import utils.visualize
import pyopenpose


def estimate(config, image, limbs_index, limbs, parts):
    if args.dump:
        dump = os.path.expanduser(os.path.expandvars(args.dump))
        tf.logging.warn('dump feature map into ' + dump)
        os.makedirs(dump, exist_ok=True)
        np.savetxt(os.path.join(dump, 'limbs_index.tsv'), limbs_index, fmt='%d', delimiter='\t')
        scipy.misc.imsave(os.path.join(dump, 'image.jpg'), image)
        np.save(os.path.join(dump, 'limbs.npy'), limbs)
        np.save(os.path.join(dump, 'parts.npy'), parts)
    threshold = config.getfloat('nms', 'threshold')
    limits = config.getint('nms', 'limits')
    steps = config.getint('integration', 'steps')
    min_score = config.getfloat('integration', 'min_score')
    min_count = config.getint('integration', 'min_count')
    cluster_min_score = config.getfloat('cluster', 'min_score')
    cluster_min_count = config.getint('cluster', 'min_count')
    if args.show == 'nms':
        utils.visualize.show_nms(image, parts, threshold, limits)
    elif args.show == 'score':
        utils.visualize.show_connection(image, limbs_index, limbs, parts, threshold, limits, steps, 0, 1)
    elif args.show == 'connection':
        utils.visualize.show_connection(image, limbs_index, limbs, parts, threshold, limits, steps, min_score, min_count)
    elif args.show == 'cluster':
        utils.visualize.show_clusters(image, limbs_index, limbs, parts, threshold, limits, steps, min_score, min_count)
    results = pyopenpose.estimate(limbs_index, limbs, parts, threshold, limits, steps, min_score, min_count, cluster_min_score, cluster_min_count)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image)
    scale_y, scale_x = utils.preprocess.calc_image_scale(parts.shape[:2], image.shape[:2])
    utils.visualize.draw_results(ax, scale_y, scale_x, results)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def read_image(path, size):
    image = utils.preprocess.read_image(path)
    image = np.array(np.uint8(image))
    return image, utils.preprocess.resize(image, size)


def eval_tensor(sess, image, _image, tensors):
    _image = _image.astype(np.float32)
    _image = utils.preprocess.per_image_standardization(_image)
    feed_dict = {image: np.expand_dims(_image, 0)}
    _tensors = sess.run(tensors, feed_dict)
    return tuple(map(operator.itemgetter(0), _tensors))


def main():
    matplotlib.rcParams.update({'font.size': args.fontsize})
    cachedir = utils.get_cachedir(config)
    logdir = utils.get_logdir(config)
    with open(cachedir + '.parts', 'r') as f:
        num_parts = int(f.read())
    limbs_index = utils.get_limbs_index(config)
    assert pyopenpose.limbs_points(limbs_index) == num_parts
    size_image = config.getint('config', 'height'), config.getint('config', 'width')
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, size_image[0], size_image[1], 3], name='image')
        net = utils.parse_attr(config.get('backbone', 'dnn'))(config, image, train=True)
        limbs, parts = utils.parse_attr(config.get('stages', 'dnn'))(config, net, len(limbs_index), num_parts)
        limbs = tf.check_numerics(limbs, limbs.op.name)
        parts = tf.check_numerics(parts[:, :, :, :-1], parts.op.name) # drop background channel
        model_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + model_path)
        slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)
        path = os.path.expanduser(os.path.expandvars(args.path))
        if os.path.isfile(path):
            image_rgb, image_resized = read_image(path, size_image[::-1])
            _limbs, _parts = eval_tensor(sess, image, image_resized, [limbs, parts])
            estimate(config, image_rgb, limbs_index, _limbs, _parts)
            plt.show()
        else:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[-1].lower() in args.exts:
                        _path = os.path.join(dirpath, filename)
                        print(_path)
                        image_rgb, image_resized = read_image(_path, size_image[::-1])
                        _limbs, _parts = eval_tensor(sess, image, image_resized, [limbs, parts])
                        estimate(config, image_rgb, limbs_index, _limbs, _parts)
                        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='input image path')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-d', '--dump', help='folder path of feature map dump file')
    parser.add_argument('-s', '--show')
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
