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

import tensorflow.contrib.slim as slim
import stages


class Stages(stages.Stages):
    def __init__(self, num_limbs, num_parts):
        super(Stages, self).__init__(num_limbs, num_parts)
        self.count = 6
    
    def stage(self, net, channels):
        if self.index == 0:
            with slim.arg_scope([slim.conv2d], num_outputs=128, kernel_size=[3, 3]):
                for index in range(3):
                    net = slim.conv2d(net, scope='conv%d' % index)
            index += 1
            net = slim.conv2d(net, 512, kernel_size=[1, 1], scope='conv%d' % index)
        else:
            with slim.arg_scope([slim.conv2d], num_outputs=128, kernel_size=[7, 7]):
                for index in range(5):
                    net = slim.conv2d(net, scope='conv%d' % index)
                index += 1
                net = slim.conv2d(net, kernel_size=[1, 1], scope='conv%d' % index)
        return slim.conv2d(net, channels, kernel_size=[1, 1], activation_fn=None)


class Original(Stages):
    def __init__(self, config, num_limbs, num_parts):
        Stages.__init__(self, num_limbs, num_parts)
