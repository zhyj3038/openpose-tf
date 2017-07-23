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

import operator
import tensorflow as tf
import tensorflow.contrib.slim as slim
import stages


class Stages(stages.Stages):
    def __init__(self, num_limbs, num_parts, stages=2, multiply=[2, 2], sqz=0, sqz0=0):
        super(Stages, self).__init__(num_limbs, num_parts, stages)
        self.multiply = multiply
        self.sqz = sqz
        self.sqz0 = sqz0
    
    def stage_branches(self, stage, _input, train):
        if stage > 0 and self.sqz0 > 1:
            channels = sum(map(operator.itemgetter(1), self.branches))
            c = max(_input.get_shape()[-1].value // self.sqz0, channels)
            _input = slim.conv2d(_input, c, kernel_size=[1, 1], scope='sqz')
        return super(Stages, self).stage_branches(stage, _input, train)
    
    def stage(self, stage, net, channels, train):
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
                    net = slim.conv2d_transpose(net, _channels)
                    _, height, width, _ = net.get_shape().as_list()
                    _, _height, _width, _ = _net.get_shape().as_list()
                    assert height <= _height
                    assert width <= _width
                    if height != _height or width != _width:
                        offsets = [0, (_height - height) // 2, (_width - width) // 2, 0]
                        size = [-1, height, width, _channels]
                        _net = tf.slice(_net, offsets, size, name='center_crop')
                    net = tf.concat([net, _net], -1)
                    if self.sqz > 1:
                        c = max(net.get_shape()[-1].value // self.sqz, channels)
                        net = slim.conv2d(net, c, kernel_size=[1, 1], scope='sqz')
                    net = slim.conv2d(net, _channels)
        return slim.conv2d(net, channels, kernel_size=[1, 1], activation_fn=None)


# 4.1M
class Unet2(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[2, 2], sqz=0, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


# M
class Unet2_1(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[2, 1.5], sqz=0, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


# 2.7M
class Unet2_2(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[1.5, 1.5], sqz=0, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


# 3.4M
class Unet2Sqz3(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[2, 2], sqz=3, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


# 13.4M
class Unet3(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[2, 2, 2], sqz=0, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


# 10.6M
class Unet3_1(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[2, 2, 1], sqz=0, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


class Unet3_2(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[2, 1.5, 1], sqz=0, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


# 10.5M
class Unet3Sqz3(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[2, 2, 2], sqz=3, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


class Unet3Sqz3_1(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[2, 2, 1], sqz=3, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)


# 6.3M
class Unet3Sqz3_2(Stages):
    def __init__(self, config, num_limbs, num_parts, stages=2, multiply=[1.9, 1.6, 1.3], sqz=3, sqz0=0):
        Stages.__init__(self, num_limbs, num_parts, stages, multiply, sqz, sqz0)
