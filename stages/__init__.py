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


def make_loss(config, net, mask, label, stage, branch, name='loss'):
    shape = net.get_shape().as_list()
    assert mask.get_shape().as_list()[:3] == shape[:3]
    assert label.get_shape().as_list() == shape
    dist = tf.square(net - label)
    loss = tf.reduce_mean(mask * dist, name=name)
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
