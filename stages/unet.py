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

import numbers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from stages import make_loss


def unet(config, net, limbs, parts, mask=None, stages=6, channels=128, sqz=128 // 3, num=2, scope='stages'):
    with tf.variable_scope(scope):
        if mask is None:
            assert isinstance(limbs, numbers.Integral)
            assert isinstance(parts, numbers.Integral)
            limbs = limbs * 2
            parts = parts + 1
        else:
            with tf.name_scope('labels'):
                mask = tf.identity(mask, 'mask')
                limbs = tf.identity(limbs, 'limbs')
                parts = tf.identity(parts, 'parts')
        branches = [('limbs', limbs), ('parts', parts)]
        
        image = tf.identity(net, 'image')
        outputs = [image]
        for stage in range(stages):
            with tf.variable_scope('stage%d' % stage):
                if len(outputs) == 1:
                    _input = tf.identity(outputs[0], 'input')
                else:
                    assert len(outputs) == len(branches)
                    _input = tf.concat(outputs + [image], -1, name='input')
                outputs = []
                for branch, label in branches:
                    net = _input
                    with tf.variable_scope(branch):
                        index = 0
                        net = tf.identity(net, 'input')
                        if stage == 0:
                            with slim.arg_scope([slim.conv2d], num_outputs=128, kernel_size=[3, 3]):
                                for _ in range(3):
                                    net = slim.conv2d(net, scope='conv%d' % index)
                                    index += 1
                            net = slim.conv2d(net, 512, kernel_size=[1, 1], scope='conv%d' % index)
                        else:
                            with slim.arg_scope([slim.conv2d], num_outputs=channels, kernel_size=[3, 3]), slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
                                net = slim.conv2d(net, scope='conv%d' % index)
                                nets = [net]
                                index += 1
                                for _ in range(num):
                                    net = slim.max_pool2d(net, scope='pool%d' % index)
                                    if sqz > 0:
                                        net = slim.conv2d(net, kernel_size=[1, 1], scope='sqz%d' % index)
                                    net = slim.conv2d(net, scope='conv%d' % index)
                                    index += 1
                                    nets.append(net)
                                nets.pop()
                                for _net in nets[::-1]:
                                    if sqz > 0:
                                        net = slim.conv2d(net, kernel_size=[1, 1], scope='sqz%d' % index)
                                    with tf.name_scope('interp%d' % index):
                                        net = tf.image.resize_images(net, _net.get_shape()[1:3])
                                        net = tf.concat([net, _net], -1)
                                    net = slim.conv2d(net, scope='conv%d' % index)
                                    index += 1
                        net = slim.conv2d(net, label if mask is None else label.get_shape().as_list()[-1], kernel_size=[1, 1], activation_fn=None, scope='conv')
                        net = tf.identity(net, 'output')
                        if mask is not None:
                            with tf.name_scope('loss') as name:
                                make_loss(config, net, mask, label, stage, branch, name)
                    outputs.append(net)
        return tuple([tf.identity(net, branch) for net, (branch, _) in zip(outputs, branches)])


def unet64(config, net, limbs, parts, mask=None, stages=6, channels=64, sqz=0, num=2, scope='stages'):
    return unet(config, net, limbs, parts, mask, stages, channels, sqz, num, scope)


def unet64_sqz3(config, net, limbs, parts, mask=None, stages=6, channels=64, sqz=64 // 3, num=2, scope='stages'):
    return unet(config, net, limbs, parts, mask, stages, channels, sqz, num, scope)
