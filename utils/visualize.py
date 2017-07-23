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
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import pyopenpose
from . import preprocess


def draw_mask(image, mask):
    _mask = scipy.misc.imresize(np.squeeze(mask), image.shape[:2])
    for y, row in enumerate(_mask):
        for x, v in enumerate(row):
            if v < 128:
                image[y, x] = 0


def draw_keypoints(ax, keypoints, limbs_index, colors=['r', 'w'], alpha=0.3):
    for _keypoints in keypoints:
        for i, (x, y, v) in enumerate(_keypoints):
            assert v >= 0
            if v > 0:
                ax.text(x, y, str(i), bbox=dict(facecolor=colors[v - 1], alpha=alpha))
        for color, (i1, i2) in zip(itertools.cycle(prop['color'] for prop in plt.rcParams['axes.prop_cycle']), limbs_index):
            x1, y1, v1 = _keypoints[i1].T
            x2, y2, v2 = _keypoints[i2].T
            if v1 > 0 and v2 > 0:
                ax.plot([x1, x2], [y1, y2], color=color)


def draw_peaks(ax, scale_y, scale_x, peaks, text, color='w', alpha=0.5):
    for y, x, _ in peaks:
        y, x = y * scale_y, x * scale_x
        ax.text(x, y, text, bbox=dict(facecolor=color, alpha=alpha), ha='center', va='center')


def draw_estimation(ax, scale_y, scale_x, clusters):
    for color, cluster in zip(itertools.cycle(prop['color'] for prop in plt.rcParams['axes.prop_cycle']), clusters):
        for (i1, y1, x1), (i2, y2, x2) in cluster:
            y1, x1 = y1 * scale_y, x1 * scale_x
            y2, x2 = y2 * scale_y, x2 * scale_x
            ax.plot([x1, x2], [y1, y2], color=color)
            ax.text(x1, y1, str(i1))
            ax.text(x2, y2, str(i2))
