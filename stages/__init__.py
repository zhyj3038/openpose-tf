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

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def make_loss(config, net, mask, label, stage, branch, name='loss'):
    shape = net.get_shape().as_list()
    assert label.get_shape().as_list() == shape
    _shape = mask.get_shape().as_list()
    assert mask.get_shape().as_list() == shape[:3]
    norm = tf.norm(net - label, axis=-1)
    loss = tf.reduce_mean(mask * norm, name=name)
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)


def original(config, net, limbs, parts, mask=None, stages=6, scope='stages'):
    with tf.variable_scope(scope):
        image = tf.identity(net, 'image')
        net = image
        if mask is None:
            assert isinstance(limbs, int)
            assert isinstance(parts, int)
        else:
            with tf.name_scope('labels'):
                mask = tf.identity(mask, 'mask')
                limbs = tf.identity(limbs, 'limbs')
                parts = tf.identity(parts, 'parts')
        branches = [('limbs', limbs), ('parts', parts)]
        
        index = 0
        with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3]):
            net = slim.layers.conv2d(net, 256, scope='conv%d' % index)
            index += 1
            net = slim.layers.conv2d(net, 128, scope='conv%d' % index)
        
        for stage in range(stages):
            with tf.variable_scope('stage%d' % stage):
                _input = tf.identity(net, 'input')
                outputs = []
                for branch, label in branches:
                    net = _input
                    with tf.variable_scope(branch):
                        index = 0
                        net = tf.identity(net, 'input')
                        if stage == 0:
                            with slim.arg_scope([slim.layers.conv2d], num_outputs=128, kernel_size=[3, 3]):
                                for _ in range(3):
                                    net = slim.layers.conv2d(net, scope='conv%d' % index)
                                    index += 1
                            net = slim.layers.conv2d(net, 512, kernel_size=[1, 1], scope='conv%d' % index)
                        else:
                            with slim.arg_scope([slim.layers.conv2d], num_outputs=128, kernel_size=[7, 7]):
                                for _ in range(5):
                                    net = slim.layers.conv2d(net, scope='conv%d' % index)
                                    index += 1
                                net = slim.layers.conv2d(net, kernel_size=[1, 1], scope='conv%d' % index)
                        channels = label if mask is None else label.get_shape().as_list()[-1]
                        net = slim.layers.conv2d(net, channels, kernel_size=[1, 1], activation_fn=None, scope='conv')
                        net = tf.identity(net, 'output')
                        if mask is not None:
                            with tf.name_scope('loss') as name:
                                make_loss(config, net, mask, label, stage, branch, name)
                    outputs.append(net)
                if stage < stages - 1:
                    net = tf.concat(outputs + [image], -1, name='output')
        limbs, parts = outputs
        return tf.identity(limbs, 'limbs'), tf.identity(parts, 'parts')
