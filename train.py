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
import shutil
import time
import inspect
import csv
import re
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import humanize
import utils.data


def summary_scalar(config):
    try:
        reduce = eval(config.get('summary', 'scalar_reduce'))
        with open(os.path.expanduser(os.path.expandvars(config.get('summary', 'scalar'))), 'r') as f:
            patterns = filter(lambda pattern: pattern[0] != '#', map(lambda line: line.strip(), f.readlines()))
        for t in utils.match_tensor(patterns):
            name = t.op.name
            if len(t.get_shape()) > 0:
                t = reduce(t)
                tf.logging.warn(name + ' is not a scalar tensor, reducing by ' + reduce.__name__)
            tf.summary.scalar(name, t)
            tf.logging.warn('add summary scalar ' + name)
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary_image(config):
    try:
        image_max = config.getint('summary', 'image_max')
        with open(os.path.expanduser(os.path.expandvars(config.get('summary', 'image_split'))), 'r') as f:
            patterns = filter(lambda pattern: pattern[0] != '#', map(lambda line: line.strip(), f.readlines()))
        for t in utils.match_tensor(patterns):
            name = t.op.name
            shape = t.get_shape().as_list()
            if len(shape) == 4:
                channels = shape[-1]
                if channels not in (1, 3, 4):
                    with tf.name_scope(name):
                        for c in range(channels):
                            _t = t[:, :, :, c:c + 1]
                            tf.summary.image('c%d' % c, _t, image_max)
                else:
                    tf.summary.image(name, t, image_max)
                tf.logging.warn('add summary image ' + name)
            else:
                tf.logging.warn('rank of %s is not 4' % name)
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary_histogram(config):
    try:
        with open(os.path.expanduser(os.path.expandvars(config.get('summary', 'histogram'))), 'r') as f:
            patterns = filter(lambda pattern: pattern[0] != '#', map(lambda line: line.strip(), f.readlines()))
        for t in utils.match_tensor(patterns):
            name = t.op.name
            tf.summary.histogram(name, t)
            tf.logging.warn('add summary histogram ' + name)
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary(config):
    summary_scalar(config)
    summary_image(config)
    summary_histogram(config)


class GradientMultipliers(list):
    def __init__(self, paths):
        for path in paths:
            path = os.path.expanduser(os.path.expandvars(path))
            with open(path, 'r') as f:
                for pattern, scale in csv.reader(f, delimiter='\t'):
                    self.append((re.compile(pattern), int(scale)))
    
    def __call__(self, name):
        for prog, scale in self:
            if prog.match(name):
                return name, scale
        raise ValueError(name + ' not found in gradient multiplier list')


def get_gradient_multipliers(paths, variables):
    gm = GradientMultipliers(paths)
    gradient_multipliers = []
    for v in variables:
        name = v.op.name
        gradient_multipliers.append(gm(name))
    return dict(gradient_multipliers)


def main():
    logdir = utils.get_logdir(config)
    if args.delete:
        tf.logging.warn('delete logging directory: ' + logdir)
        shutil.rmtree(logdir, ignore_errors=True)
    cachedir = utils.get_cachedir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    height, width = config.getint('config', 'height'), config.getint('config', 'width')
    feature_height, feature_width = utils.calc_downsampling_size(config.get('backbone', 'dnn'), height, width)
    paths = [os.path.join(cachedir, phase + '.tfrecord') for phase in args.phase]
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
    tf.logging.warn('num_examples=%d' % num_examples)
    with tf.name_scope('batch'):
        image, mask, _, limbs, parts = utils.data.load_data(config, paths, height, width, feature_height, feature_width, num_parts, limbs_index)
        with tf.name_scope('per_image_standardization'):
            image = tf.image.per_image_standardization(image)
        batch = tf.train.shuffle_batch([image, mask, limbs, parts], batch_size=args.batch_size,
            capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count()
        )
        image, mask, limbs, parts = batch
        with tf.name_scope('output'):
            image, mask, limbs, parts = tf.identity(image, 'image'), tf.identity(mask, 'mask'), tf.identity(limbs, 'limbs'), tf.identity(parts, 'parts')
    net = utils.parse_attr(config.get('backbone', 'dnn'))(config, image, train=True)
    assert tuple(net.get_shape().as_list()[1:3]) == (feature_height, feature_width), str(net.get_shape().as_list()[1:3]) + ' != ' + str([feature_height, feature_width])
    with tf.variable_scope('stages'):
        stages = utils.parse_attr(config.get('stages', 'dnn'))(config, len(limbs_index), num_parts)
        stages.train = True
        try:
            count = config.getint('stages', 'count')
        except configparser.NoOptionError:
            count = stages.count
        for _ in range(count):
            stages(net)
    with tf.name_scope('loss_stages'):
        stages.loss(mask, limbs, parts)
    tf.logging.warn(humanize.naturalsize(sum(np.multiply.reduce(var.get_shape().as_list()) * var.dtype.size for var in tf.global_variables())))
    with tf.name_scope('total_loss') as name:
        total_loss = tf.losses.get_total_loss(name=name)
    global_step = tf.contrib.framework.get_or_create_global_step()
    try:
        gradient_multipliers = get_gradient_multipliers([config.get('backbone', 'gradient_multipliers'), config.get('stages', 'gradient_multipliers')], tf.trainable_variables())
    except configparser.NoOptionError:
        tf.logging.warn('gradient_multipliers disabled')
        gradient_multipliers = None
    with tf.variable_scope('optimizer'):
        try:
            decay_steps = config.getint('exponential_decay', 'decay_steps')
            decay_rate = config.getfloat('exponential_decay', 'decay_rate')
            staircase = config.getboolean('exponential_decay', 'staircase')
            learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps, decay_rate, staircase=staircase)
            tf.logging.warn('using a learning rate start from %f with exponential decay (decay_steps=%d, decay_rate=%f, staircase=%d)' % (args.learning_rate, decay_steps, decay_rate, staircase))
        except (configparser.NoSectionError, configparser.NoOptionError):
            learning_rate = args.learning_rate
            tf.logging.warn('using a staionary learning rate %f' % args.learning_rate)
        optimizer = utils.get_optimizer(config, args.optimizer)(learning_rate)
        tf.logging.warn('optimizer=' + args.optimizer)
        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step,
            clip_gradient_norm=args.gradient_clip, summarize_gradients=config.getboolean('summary', 'gradients'),
            gradient_multipliers=gradient_multipliers
        )
    variables_to_restore = slim.get_variables_to_restore(exclude=args.exclude)
    if args.transfer:
        path = os.path.expanduser(os.path.expandvars(args.transfer))
        tf.logging.warn('transferring from ' + path)
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(path, variables_to_restore)
        def init_fn(sess):
            sess.run(init_assign_op, init_feed_dict)
            tf.logging.warn('transferring from global_step=%d, learning_rate=%f' % sess.run((global_step, learning_rate)))
    else:
        init_fn = lambda sess: tf.logging.warn('global_step=%d, learning_rate=%f' % sess.run((global_step, learning_rate)))
    summary(config)
    tf.logging.warn('tensorboard --logdir ' + logdir)
    slim.learning.train(train_op, logdir, master=args.master, is_chief=(args.task == 0),
        global_step=global_step, number_of_steps=args.steps, init_fn=init_fn,
        summary_writer=tf.summary.FileWriter(os.path.join(logdir, args.logname)),
        save_summaries_secs=args.summary_secs, save_interval_secs=args.save_secs
    )


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-t', '--transfer', help='transferring model from a .ckpt file')
    parser.add_argument('-e', '--exclude', nargs='+', help='exclude variables while transferring')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val'])
    parser.add_argument('-s', '--steps', type=int, default=None, help='max number of steps')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('-o', '--optimizer', default='adam')
    parser.add_argument('-n', '--logname', default=time.strftime('%Y-%m-%d_%H-%M-%S'), help='the name for TensorBoard')
    parser.add_argument('-g', '--gradient_clip', default=0, type=float, help='gradient clip')
    parser.add_argument('-lr', '--learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--summary_secs', default=30, type=int, help='seconds to save summaries')
    parser.add_argument('--save_secs', default=600, type=int, help='seconds to save model')
    parser.add_argument('--level', help='logging level')
    parser.add_argument('--master', default='', help='master address')
    parser.add_argument('--task', type=int, default=0, help='task ID')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
