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
import time
import itertools
import pybenchmark
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.colors
import cv2
import pyopenpose
import utils.preprocess


@pybenchmark.profile("eval")
def eval_tensor(sess, image, _image, tensors):
    _image = _image.astype(np.float32)
    _image = utils.preprocess.per_image_standardization(_image)
    feed_dict = {image: np.expand_dims(_image, 0)}
    _tensors = sess.run(tensors, feed_dict)
    return tuple(map(operator.itemgetter(0), _tensors))


def main():
    cachedir = utils.get_cachedir(config)
    logdir = utils.get_logdir(config)
    with open(cachedir + '.parts', 'r') as f:
        num_parts = int(f.read())
    limbs_index = utils.get_limbs_index(config)
    assert pyopenpose.limbs_points(limbs_index) == num_parts
    size_image = config.getint('config', 'height'), config.getint('config', 'width')
    
    threshold = config.getfloat('nms', 'threshold')
    limits = config.getint('nms', 'limits')
    steps = config.getint('integration', 'steps')
    min_score = config.getfloat('integration', 'min_score')
    min_count = config.getint('integration', 'min_count')
    cluster_min_score = config.getfloat('cluster', 'min_score')
    cluster_min_count = config.getint('cluster', 'min_count')
    colors = itertools.cycle(matplotlib.colors.hex2color(prop['color']) for prop in plt.rcParams['axes.prop_cycle'])
    
    @pybenchmark.profile("estimate")
    def _estimate(image, limbs, parts):
        results = pyopenpose.estimate(limbs_index, limbs, parts, threshold, limits, steps, min_score, min_count, cluster_min_score, cluster_min_count)
        scale_y, scale_x = utils.preprocess.calc_image_scale(parts.shape[:2], image.shape[:2])
        for color, keypoints in zip(colors, results):
            for (y1, x1), (y2, x2) in keypoints:
                y1, x1 = int(y1 * scale_y), int(x1 * scale_x)
                y2, x2 = int(y2 * scale_y), int(x2 * scale_x)
                cv2.line(image, (x1, y1), (x2, y2), color, 3)
    
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, size_image[0], size_image[1], 3], name='image')
        net = utils.parse_attr(config.get('backbone', 'dnn'))(config, image, train=True)
        limbs, parts = utils.parse_attr(config.get('stages', 'dnn'))(config, net, len(limbs_index), num_parts)
        limbs = tf.check_numerics(limbs, limbs.op.name)
        parts = tf.check_numerics(parts[:, :, :, :-1], parts.op.name) # drop background channel
        model_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + model_path)
        slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, image_bgr = cap.read()
                assert ret
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_resized = utils.preprocess.resize(image_rgb, size_image)
                _limbs, _parts = eval_tensor(sess, image, image_resized, [limbs, parts])
                _estimate(image_bgr, _limbs, _parts)
                cv2.imshow('estimation', image_bgr)
                cv2.waitKey(1)
        except KeyboardInterrupt:
            if args.dump:
                dump = os.path.expanduser(os.path.expandvars(args.dump))
                path = os.path.join(dump, time.strftime(args.format))
                scipy.misc.imsave(path, image_rgb)
                tf.logging.warn('image dumped into ' + path)
        finally:
            cv2.destroyAllWindows()
            cap.release()
            tf.logging.info(pybenchmark.stats)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-d', '--dump', help='dump directory')
    parser.add_argument('-f', '--format', default='%Y-%m-%d_%H-%M-%S.jpg', help='dump file name format')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()