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
import humanize
import cv2
import pyopenpose
import utils.preprocess


@pybenchmark.profile('dnn')
def eval_tensor(sess, image, _image, tensors):
    _tensors = sess.run(tensors, {image: np.expand_dims(_image, 0)})
    return tuple(map(operator.itemgetter(0), _tensors))


def main():
    logdir = utils.get_logdir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    height, width = config.getint('config', 'height'), config.getint('config', 'width')
    
    threshold = config.getfloat('nms', 'threshold')
    radius = config.getint('nms', 'radius')
    steps = config.getint('integration', 'steps')
    min_score = config.getfloat('integration', 'min_score')
    min_count = config.getint('integration', 'min_count')
    cluster_min_score = config.getfloat('cluster', 'min_score')
    cluster_min_count = config.getint('cluster', 'min_count')
    colors = [tuple(map(lambda c: c * 255, matplotlib.colors.colorConverter.to_rgb(prop['color']))) for prop in plt.rcParams['axes.prop_cycle']]
    
    def _estimate(image, limbs, parts, font=cv2.FONT_HERSHEY_SIMPLEX):
        clusters = pybenchmark.profile('estimate')(pyopenpose.estimate)(limbs_index, limbs, parts, threshold, radius, steps, min_score, min_count, cluster_min_score, cluster_min_count)
        scale_y, scale_x = utils.preprocess.calc_image_scale(parts.shape[:2], image.shape[:2])
        for color, cluster in zip(itertools.cycle(colors), clusters):
            for (i1, y1, x1), (i2, y2, x2) in cluster:
                y1, x1 = int(y1 * scale_y), int(x1 * scale_x)
                y2, x2 = int(y2 * scale_y), int(x2 * scale_x)
                cv2.line(image, (x1, y1), (x2, y2), color, 3)
                if args.part:
                    cv2.putText(image, str(i1), (x1, y1), font, 1, (255,255,255), 2)
                    cv2.putText(image, str(i2), (x2, y2), font, 1, (255,255,255), 2)
    
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, height, width, 3], name='image')
        net = utils.parse_attr(config.get('backbone', 'dnn'))(config, image, train=True)
        limbs, parts = utils.parse_attr(config.get('stages', 'dnn'))(config, net, len(limbs_index), num_parts)
        limbs = tf.check_numerics(limbs, limbs.op.name)
        parts = tf.check_numerics(parts[:, :, :, :-1], parts.op.name) # drop background channel
        tf.logging.info(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
        tf.logging.info('locating checkpoint in ' + logdir)
        checkpoint_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + checkpoint_path)
        slim.assign_from_checkpoint_fn(checkpoint_path, tf.global_variables())(sess)
        cap = cv2.VideoCapture(args.camera)
        try:
            while True:
                ret, image_bgr = cap.read()
                assert ret
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_resized = utils.preprocess.resize(image_rgb, height, width)
                _limbs, _parts = eval_tensor(sess, image, utils.preprocess.per_image_standardization(image_resized.astype(np.float32)), [limbs, parts])
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
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('-d', '--dump', help='dump directory')
    parser.add_argument('-p', '--part', action='store_true', help='show part numbers')
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