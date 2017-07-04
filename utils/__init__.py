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


def get_cachedir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('cache', 'basedir')))
    dataset = os.path.basename(config.get('cache', 'dataset'))
    return os.path.join(basedir, 'cache', dataset)


def get_logdir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('cache', 'basedir')))
    dataset = os.path.basename(config.get('cache', 'dataset'))
    backbone = os.path.splitext(config.get('backbone', 'dnn'))[-1][1:]
    stages = os.path.splitext(config.get('stages', 'dnn'))[-1][1:]
    return os.path.join(basedir, dataset, backbone, stages)


def get_limbs_index(config):
    dataset = os.path.expanduser(os.path.expandvars(config.get('cache', 'dataset')))
    limbs_index = np.loadtxt(dataset + '.tsv', dtype=np.int, delimiter='\t')
    assert pyopenpose.limbs_points(limbs_index) == get_dataset_mappers(config)[1]
    return limbs_index


def parse_attr(s):
    module, name = s.rsplit('.', 1)
    module = importlib.import_module(module)
    return getattr(module, name)


def calc_downsampling_size(dnn, height, width):
    func = parse_attr(dnn + '_downsampling')
    return func(height, width)


def match_trainable_variables(pattern):
    prog = re.compile(pattern)
    return [v for v in tf.trainable_variables() if prog.match(v.op.name)]


def match_tensor(pattern):
    prog = re.compile(pattern)
    return [op.values()[0] for op in tf.get_default_graph().get_operations() if op.values() and prog.match(op.name)]


def load_config(config, paths):
    for path in paths:
        path = os.path.expanduser(os.path.expandvars(path))
        assert os.path.exists(path)
        config.read(path)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
