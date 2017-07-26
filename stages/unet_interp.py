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
from . import unet


class Stages(unet.Stages):
    def __init__(self, num_limbs, num_parts):
        super(Stages, self).__init__(num_limbs, num_parts)
    
    def stage(self, net, channels):
        _channels = channels
        with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]), slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]), slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2, padding='VALID'):
            nets = []
            for index, multiply in enumerate(self.multiply):
                with tf.variable_scope('down%d' % index):
                    net = slim.conv2d(net, _channels)
                    nets.append((index, net))
                    net = slim.max_pool2d(net)
                    _channels = int(_channels * multiply)
            net = slim.conv2d(net, _channels, scope='conv%d' % len(nets))
            for index, _net in nets[::-1]:
                with tf.variable_scope('up%d' % index):
                    _channels = _net.get_shape()[-1].value
                    with tf.name_scope('interp%d' % index):
                        net = tf.image.resize_images(net, _net.get_shape()[1:3])
                    net = tf.concat([net, _net], -1)
                    c = net.get_shape()[-1].value
                    for sqz in self.sqz:
                        net = slim.conv2d(net, max(int(c * sqz), channels), kernel_size=[1, 1], scope='sqz')
                    net = slim.conv2d(net, _channels)
        return slim.conv2d(net, channels, kernel_size=[1, 1], activation_fn=None)


# 3.1M
class Unet2Sqz3(Stages):
    def __init__(self, config, num_limbs, num_parts):
        Stages.__init__(self, num_limbs, num_parts)
        self.sqz = [1 / 3]


# M
class Unet2Sqz3_1(Stages):
    def __init__(self, config, num_limbs, num_parts):
        Stages.__init__(self, num_limbs, num_parts)
        self.multiply = [1.5, 1.2]
        self.sqz = [1 / 3]
