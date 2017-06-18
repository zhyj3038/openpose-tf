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


def make_loss(config, limbs, net, mask, label, stage, name='loss'):
    shape = net.get_shape().as_list()
    assert shape == label.get_shape().as_list()
    _shape = mask.get_shape().as_list()
    assert shape[:-1] == _shape[:-1] and _shape[-1] == 1
    diff = tf.square(net - label, name='diff')
    cnt = np.multiply.reduce(_shape[:-1])
    loss = tf.identity(tf.reduce_sum(mask * diff) / cnt, name)
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)


def original(config, limbs, parts, net, mask=None, label=None, stages=6, scope='stages'):
    branches = [('limbs', limbs * 2), ('parts', parts + 1)]
    assert label is None or sum(map(lambda item: item[-1], branches)) == label.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        image = tf.identity(net, 'image')
        net = image
        
        index = 0
        with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3]):
            net = slim.layers.conv2d(net, 256, scope='conv%d' % index)
            index += 1
            net = slim.layers.conv2d(net, 128, scope='conv%d' % index)
        
        for stage in range(stages):
            with tf.variable_scope('stage%d' % stage):
                inputs = tf.identity(net, 'inputs')
                outputs = []
                if stage == 0:
                    for branch, channels in branches:
                        net = inputs
                        with tf.variable_scope(branch) as name:
                            index = 0
                            with slim.arg_scope([slim.layers.conv2d], num_outputs=128, kernel_size=[3, 3]):
                                for _ in range(3):
                                    net = slim.layers.conv2d(net, scope='conv%d' % index)
                                    index += 1
                            net = slim.layers.conv2d(net, 512, kernel_size=[1, 1], scope='conv%d' % index)
                            net = slim.layers.conv2d(net, channels, kernel_size=[1, 1], activation_fn=None, scope=name)
                            outputs.append(net)
                else:
                    for branch, channels in branches:
                        net = inputs
                        with tf.variable_scope(branch) as name:
                            index = 0
                            with slim.arg_scope([slim.layers.conv2d], num_outputs=128, kernel_size=[7, 7]):
                                for _ in range(5):
                                    net = slim.layers.conv2d(net, scope='conv%d' % index)
                                    index += 1
                                net = slim.layers.conv2d(net, kernel_size=[1, 1], scope='conv%d' % index)
                            net = slim.layers.conv2d(net, channels, kernel_size=[1, 1], activation_fn=None, scope=name)
                            outputs.append(net)
                if mask is not None and label is not None:
                    with tf.name_scope('loss') as name:
                        make_loss(config, limbs, tf.concat(outputs, -1, name='result'), mask, label, stage, name)
                if stage < stages - 1:
                    net = tf.concat(outputs + [image], -1, name='output')
        limbs, parts = outputs
        return tf.identity(limbs, 'limbs'), tf.identity(parts, 'parts')
