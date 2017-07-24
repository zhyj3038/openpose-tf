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


class Stages(list):
    def __init__(self, num_limbs, num_parts):
        self.branches = [('limbs', num_limbs * 2), ('parts', num_parts + 1)]
        self.train = False
        self.outputs = []
        self.index = 0
    
    def __call__(self, net):
        if not self.outputs:
            self.image = tf.identity(net, 'image')
            self.outputs = [self.image]
        with tf.variable_scope('stage%d' % self.index):
            if len(self.outputs) == 1:
                _input = tf.identity(self.outputs[0], 'input')
            else:
                assert len(self.outputs) == len(self.branches)
                _input = tf.concat(self.outputs + [self.image], -1, name='input')
            self.outputs = self.stage_branches(_input)
            self.append(self.outputs)
            self.index += 1
        return self.outputs
    
    def stage_branches(self, net):
        outputs = []
        for branch, channels in self.branches:
            outputs.append(self.stage_branch(net, channels, branch))
        return outputs
    
    def stage_branch(self, net, channels, scope):
        with tf.variable_scope(scope):
            net = tf.identity(net, 'input')
            net = self.stage(net, channels)
            return tf.identity(net, 'output')
    
    def stage(self, net, channels):
        pass
    
    def loss(self, mask, limbs, parts):
        with tf.name_scope('labels'):
            mask = tf.identity(mask, 'mask')
            limbs = tf.identity(limbs, 'limbs')
            parts = tf.identity(parts, 'parts')
        for stage, (_limbs, _parts) in enumerate(self):
            with tf.variable_scope('stage%d' % stage):
                make_loss(_limbs, mask, limbs, 'limbs')
                make_loss(_parts, mask, parts, 'parts')


def make_loss(net, mask, label, scope=None):
    if scope is None:
        scope = inspect.stack()[0][3]
    with tf.name_scope(scope) as name:
        shape = net.get_shape().as_list()
        assert mask.get_shape().as_list()[:3] == shape[:3], str(mask.get_shape().as_list()[:3]) + ' != ' + str(shape[:3])
        assert label.get_shape().as_list() == shape
        dist = tf.square(net - label)
        loss = tf.reduce_mean(mask * dist, name=name)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
