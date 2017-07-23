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

import tensorflow as tf
import tensorflow.contrib.slim as slim


def vgg19_10(config, inputs, train=False, shrink=1):
    with tf.variable_scope('vgg_19'):
        with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]), slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64 // shrink, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128 // shrink, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256 // shrink, scope='conv3')
            net = slim.max_pool2d(net, scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512 // shrink, scope='conv4')
    with tf.variable_scope('backbone'):
        index = 0
        with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
            net = slim.conv2d(net, 256 // shrink, scope='conv%d' % index)
            index += 1
            net = slim.conv2d(net, 128 // shrink, scope='conv%d' % index)
    return net


def vgg19_10_downsampling(height, width):
    return height // 2 ** 3, width // 2 ** 3
