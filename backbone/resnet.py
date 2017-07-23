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

import inspect
import tensorflow as tf
import tensorflow.contrib.slim as slim


@slim.add_arg_scope
def unit2(net, depth, bottleneck, stride):
    net = slim.layers.conv2d(net, bottleneck, [3, 3], stride, scope='bottleneck')
    return slim.layers.conv2d(net, depth, [3, 3], activation_fn=None)


@slim.add_arg_scope
def unit3(net, depth, bottleneck, stride):
    net = slim.layers.conv2d(net, bottleneck, [1, 1], stride, scope='bottleneck1x1')
    net = slim.layers.conv2d(net, bottleneck, [3, 3], scope='bottleneck')
    return slim.layers.conv2d(net, depth, [1, 1], activation_fn=None)


def _resnet(net, train, scope, shrink, blocks, unit):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.batch_norm], scale=True, is_training=train), slim.arg_scope([slim.layers.conv2d], normalizer_fn=slim.batch_norm):
            net = slim.layers.conv2d(net, 64 // shrink, [7, 7], 2)
            net = slim.layers.max_pool2d(net, [3, 3])
            for b, (units, depth, bottleneck, stride) in enumerate(blocks):
                with tf.variable_scope('block%d' % b):
                    for u in range(units):
                        with tf.variable_scope('unit%d' % u):
                            _input = net = tf.identity(net, 'input')
                            net = unit(net, depth, bottleneck, 1 if u > 0 else stride)
                            shape = net.get_shape().as_list()
                            _shape = _input.get_shape().as_list()
                            if shape == _shape:
                                shortcut = _input
                            elif shape[-1] == _shape[-1]:
                                shortcut = slim.layers.max_pool2d(_input, [1, 1], stride, scope='shortcut')
                            else:
                                shortcut = slim.layers.conv2d(_input, depth, [1, 1], stride, activation_fn=None, scope='shortcut')
                            net = tf.nn.relu(shortcut + net)
    return net


def resnet_openpose(config, net, train=False, scope=None, shrink=2):
    if scope is None:
        scope = inspect.stack()[0][3]
    blocks = [
        (2, 64 // shrink, 64 // shrink, 1),
        (2, 128 // shrink, 128 // shrink, 2),
    ]
    return _resnet(net, train, scope, shrink, blocks, unit2)


def resnet_openpose_downsampling(height, width):
    return height // 2 ** 3, width // 2 ** 3


def resnet18(net, train=False, scope=None, shrink=1):
    if scope is None:
        scope = inspect.stack()[0][3]
    blocks = [
        (2, 64 // shrink, 64 // shrink, 1),
        (2, 128 // shrink, 128 // shrink, 2),
        (2, 256 // shrink, 256 // shrink, 2),
        (2, 512 // shrink, 512 // shrink, 2),
    ]
    return _resnet(net, train, scope, shrink, blocks, unit2)


def resnet18_4(net, train=False, scope=None):
    if scope is None:
        scope = inspect.stack()[0][3]
    return resnet18(net, train, scope, 4)


def resnet34(net, train=False, scope=None, shrink=1):
    if scope is None:
        scope = inspect.stack()[0][3]
    blocks = [
        (3, 64 // shrink, 64 // shrink, 1),
        (4, 128 // shrink, 128 // shrink, 2),
        (6, 256 // shrink, 256 // shrink, 2),
        (3, 512 // shrink, 512 // shrink, 2),
    ]
    return _resnet(net, train, scope, shrink, blocks, unit2)


def resnet50(net, train=False, scope=None, shrink=1):
    if scope is None:
        scope = inspect.stack()[0][3]
    blocks = [
        (3, 256 // shrink, 64 // shrink, 1),
        (4, 512 // shrink, 128 // shrink, 2),
        (6, 1024 // shrink, 256 // shrink, 2),
        (3, 2048 // shrink, 512 // shrink, 2),
    ]
    return _resnet(net, train, scope, shrink, blocks, unit3)


def resnet101(net, train=False, scope=None, shrink=1):
    if scope is None:
        scope = inspect.stack()[0][3]
    blocks = [
        (3, 256 // shrink, 64 // shrink, 1),
        (4, 512 // shrink, 128 // shrink, 2),
        (23, 1024 // shrink, 256 // shrink, 2),
        (3, 2048 // shrink, 512 // shrink, 2),
    ]
    return _resnet(net, train, scope, shrink, blocks, unit3)


def resnet152(net, train=False, scope=None, shrink=1):
    blocks = [
        (3, 256 // shrink, 64 // shrink, 1),
        (8, 512 // shrink, 128 // shrink, 2),
        (36, 1024 // shrink, 256 // shrink, 2),
        (3, 2048 // shrink, 512 // shrink, 2),
    ]
    return _resnet(net, train, scope, shrink, blocks, unit3)
