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

import itertools
import matplotlib.pyplot as plt


def draw_mask(image, mask):
    for y, row in enumerate(mask):
        for x, v in enumerate(row):
            if not v:
                image[y, x] = 0


def draw_keypoints(ax, keypoints, limbs, colors=['r', 'w'], alpha=0.3):
    limb_colors = [prop['color'] for _, prop in zip(limbs, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    for _keypoints in keypoints:
        for i, (x, y, v) in enumerate(_keypoints):
            assert v >= 0
            if v > 0:
                plt.text(x, y, str(i), bbox=dict(facecolor=colors[v - 1], alpha=alpha))
        for i, (i1, i2) in enumerate(limbs):
            x1, y1, v1 = _keypoints[i1].T
            x2, y2, v2 = _keypoints[i2].T
            if v1 > 0 and v2 > 0:
                plt.plot([x1, x2], [y1, y2], color=limb_colors[i])
