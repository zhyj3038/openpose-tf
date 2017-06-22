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


def draw_mask(image, mask):
    _mask = scipy.misc.imresize(np.squeeze(mask), image.shape[:2])
    for y, row in enumerate(_mask):
        for x, v in enumerate(row):
            if v < 128:
                image[y, x] = 0


def draw_keypoints(ax, keypoints, limbs_index, colors=['r', 'w'], alpha=0.3):
    limb_colors = [prop['color'] for _, prop in zip(limbs_index, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    for _keypoints in keypoints:
        for i, (x, y, v) in enumerate(_keypoints):
            assert v >= 0
            if v > 0:
                ax.text(x, y, str(i), bbox=dict(facecolor=colors[v - 1], alpha=alpha))
        for i, (i1, i2) in enumerate(limbs_index):
            x1, y1, v1 = _keypoints[i1].T
            x2, y2, v2 = _keypoints[i2].T
            if v1 > 0 and v2 > 0:
                ax.plot([x1, x2], [y1, y2], color=limb_colors[i])


def draw_nms(ax, height, width, parts, threshold, limits, alpha=0.5):
    _height, _width, _ = parts.shape
    peaks = pyopenpose.featuremap_peaks(parts, threshold, limits)
    for i, _peaks in enumerate(peaks):
        for y, x, _ in _peaks:
            ax.text(x * width / _width, y * height / _height, str(i), bbox=dict(facecolor='w', alpha=alpha))


def draw_score(ax, height, width, limbs_index, limbs, parts, threshold, limits, steps, min_score, min_count):
    _height, _width, _ = limbs.shape
    peaks = pyopenpose.featuremap_peaks(parts, threshold, limits)
    limbs = np.reshape(limbs, [_height, _width, -1, 2])
    limbs = np.transpose(limbs, [2, 3, 0, 1])
    limb_colors = [prop['color'] for _, prop in zip(limbs_index, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    for i, ((i1, i2), (limb_x, limb_y), color) in enumerate(zip(limbs_index, limbs, limb_colors)):
        peaks1 = peaks[i1]
        peaks2 = peaks[i2]
        connections = pyopenpose.calc_limb_score(peaks1, peaks2, limb_x, limb_y, steps, min_score, min_count)
        for p1, p2, score in connections:
            y1, x1, v1 = peaks1[p1]
            y2, x2, v2 = peaks2[p2]
            ax.plot([x1 * width / _width, x2 * width / _width], [y1 * height / _height, y2 * height / _height], color=color)
