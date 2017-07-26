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
import re
import csv
import importlib
import inspect
import numpy as np
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.python.client import device_lib
import pyopenpose


class DatasetMapper(object):
    def __init__(self, size, mapper):
        self.size = size
        self.mapper = mapper
    
    def __call__(self, parts, dtype=np.int64):
        assert len(parts.shape) == 2 and parts.shape[-1] == 3
        result = np.zeros([self.size, 3], dtype=parts.dtype)
        for i, func in self.mapper:
            result[i] = func(parts)
        return result


def get_dataset_mappers(config):
    dataset = os.path.expanduser(os.path.expandvars(config.get('cache', 'dataset')))
    mappers = {}
    for filename in os.listdir(dataset):
        path = os.path.join(dataset, filename)
        if os.path.isfile(path) and os.path.splitext(filename)[-1].lower() == '.tsv':
            dataset = os.path.splitext(filename)[0]
            with open(path, 'r') as f:
                mapper = [(int(s), eval(func)) for s, func in csv.reader(f, delimiter='\t')]
            mappers[dataset] = mapper
    size = max(map(lambda mapper: max(map(lambda item: item[0], mapper)), mappers.values())) + 1
    for dataset in mappers:
        mappers[dataset] = DatasetMapper(size, mappers[dataset])
    return mappers, size


def get_symmetric_parts(config):
    dataset = os.path.expanduser(os.path.expandvars(config.get('cache', 'dataset')))
    with open(dataset + '.txt', 'r') as f:
        symmetric_parts = [list(map(int, line.rstrip().split())) for line in f]
    for i, symmetric in enumerate(symmetric_parts):
        for j in symmetric:
            assert i != j
    assert len(symmetric_parts) == get_dataset_mappers(config)[1]
    return symmetric_parts


def get_limbs_index(config):
    dataset = os.path.expanduser(os.path.expandvars(config.get('cache', 'dataset')))
    limbs_index = np.loadtxt(dataset + '.tsv', dtype=np.int, delimiter='\t')
    assert pyopenpose.limbs_points(limbs_index) == get_dataset_mappers(config)[1]
    return limbs_index


def get_cachedir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('cache', 'basedir')))
    dataset = os.path.basename(config.get('cache', 'dataset'))
    return os.path.join(basedir, 'cache', dataset)


def get_logdir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('cache', 'basedir')))
    dataset = os.path.basename(config.get('cache', 'dataset'))
    backbone = config.get('backbone', 'dnn')
    stages = config.get('stages', 'dnn')
    return os.path.join(basedir, 'logdir', dataset, backbone, stages)


def parse_attr(s):
    module, name = s.rsplit('.', 1)
    module = importlib.import_module(module)
    return getattr(module, name)


def calc_downsampling_size(dnn, height, width):
    func = parse_attr(dnn + '_downsampling')
    return func(height, width)


class MultiRegex(list):
    def __init__(self, patterns):
        for pattern in patterns:
            if pattern:
                self.append(re.compile(pattern))
    
    def match(self, s):
        for prog in self:
            m = prog.match(s)
            if m:
                return m
        return None


def match_trainable_variables(patterns):
    prog = MultiRegex(patterns)
    return [v for v in tf.trainable_variables() if prog.match(v.op.name)]


def match_tensor(patterns):
    prog = MultiRegex(patterns)
    return [op.values()[0] for op in tf.get_default_graph().get_operations() if op.values() and prog.match(op.name)]


def get_optimizer(config, name):
    section = 'optimizer_' + name
    return {
        'adam': lambda learning_rate: tf.train.AdamOptimizer(learning_rate, config.getfloat(section, 'beta1'), config.getfloat(section, 'beta2'), config.getfloat(section, 'epsilon')),
        'adadelta': lambda learning_rate: tf.train.AdadeltaOptimizer(learning_rate, config.getfloat(section, 'rho'), config.getfloat(section, 'epsilon')),
        'adagrad': lambda learning_rate: tf.train.AdagradOptimizer(learning_rate, config.getfloat(section, 'initial_accumulator_value')),
        'momentum': lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, config.getfloat(section, 'momentum')),
        'rmsprop': lambda learning_rate: tf.train.RMSPropOptimizer(learning_rate, config.getfloat(section, 'decay'), config.getfloat(section, 'momentum'), config.getfloat(section, 'epsilon')),
        'ftrl': lambda learning_rate: tf.train.FtrlOptimizer(learning_rate, config.getfloat(section, 'learning_rate_power'), config.getfloat(section, 'initial_accumulator_value'), config.getfloat(section, 'l1_regularization_strength'), config.getfloat(section, 'l2_regularization_strength')),
        'gd': lambda learning_rate: tf.train.GradientDescentOptimizer(learning_rate),
    }[name]


def load_config(config, paths):
    for path in paths:
        path = os.path.expanduser(os.path.expandvars(path))
        assert os.path.exists(path)
        config.read(path)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
