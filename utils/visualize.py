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


def draw_peaks(ax, scale_y, scale_x, peaks, text, color='w', alpha=0.5):
    for y, x, _ in peaks:
        y, x = y * scale_y, x * scale_x
        ax.text(x, y, text, bbox=dict(facecolor=color, alpha=alpha), ha='center', va='center')


def draw_results(ax, scale_y, scale_x, results):
    for prop, keypoints in zip(itertools.cycle(plt.rcParams['axes.prop_cycle']), results):
        for (y1, x1), (y2, x2) in keypoints:
            y1, x1 = y1 * scale_y, x1 * scale_x
            y2, x2 = y2 * scale_y, x2 * scale_x
            ax.plot([x1, x2], [y1, y2], color=prop['color'])


def show_nms(image, parts, threshold, limits, alpha=0.5):
    height, width, _ = image.shape
    _height, _width, _ = parts.shape
    scale_y, scale_x = height / _height, width / _width
    for index, feature in enumerate(np.transpose(parts, [2, 0, 1])):
        peaks = pyopenpose.feature_peaks(feature, threshold, limits)
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(image)
        ax.imshow(scipy.misc.imresize(feature, (height, width)), alpha=alpha)
        draw_peaks(ax, scale_y, scale_x, peaks, str(index))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.set_window_title('part%d' % index)
        fig.tight_layout()
        plt.show()
    peaks = pyopenpose.featuremap_peaks(parts, threshold, limits)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image)
    for index, _peaks in enumerate(peaks):
        draw_peaks(ax, scale_y, scale_x, _peaks, str(index))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def show_connection(image, limbs_index, limbs, parts, threshold, limits, steps, min_score, min_count, alpha=0.5, linewidth=10):
    height, width, _ = image.shape
    _height, _width, _ = limbs.shape
    scale_y, scale_x = height / _height, width / _width
    peaks = pyopenpose.featuremap_peaks(parts, threshold, limits)
    assert len(peaks) == parts.shape[-1]
    _limbs = np.reshape(limbs, [_height, _width, -1, 2])
    _limbs = np.transpose(_limbs, [2, 3, 0, 1])
    limb_colors = [prop['color'] for _, prop in zip(limbs_index, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    for index, ((i1, i2), (limb_x, limb_y), color) in enumerate(zip(limbs_index, _limbs, limb_colors)):
        fig, axes = plt.subplots(1, 3)
        for ax in axes.flat:
            ax.imshow(image)
        axes.flat[0].imshow(scipy.misc.imresize(limb_x, (height, width)), alpha=alpha)
        axes.flat[1].imshow(scipy.misc.imresize(limb_y, (height, width)), alpha=alpha)
        for i, _peaks in enumerate(peaks):
            for ax in axes.flat[:2]:
                for j, (y, x, _) in enumerate(_peaks):
                    y, x = y * scale_y, x * scale_x
                    ax.text(x, y, '%d_%d' % (i, j) if i in (i1, i2) else str(i), bbox=dict(facecolor='r' if i == i1 else 'g' if i == i2 else 'w', alpha=alpha), ha='center', va='center')
        peaks1 = peaks[i1]
        peaks2 = peaks[i2]
        connections = pyopenpose.calc_limb_score(index * 2, limbs, peaks1, peaks2, steps, min_score, min_count)
        for ax in axes.flat[:2]:
            for p1, p2, score in connections:
                y1, x1, _ = peaks1[p1]
                y2, x2, _ = peaks2[p2]
                y1, x1 = y1 * scale_y, x1 * scale_x
                y2, x2 = y2 * scale_y, x2 * scale_x
                ax.plot([x1, x2], [y1, y2])
        if min_score > 0 and min_count > 0:
            pyopenpose.filter_connections(connections, len(peaks1), len(peaks2))
        max_score = max(connections, key=lambda item: item[-1])[-1] if connections else 0
        ax = axes.flat[-1]
        for p1, p2, score in connections:
            y1, x1, _ = peaks1[p1]
            y2, x2, _ = peaks2[p2]
            y1, x1 = y1 * scale_y, x1 * scale_x
            y2, x2 = y2 * scale_y, x2 * scale_x
            ax.text(x1, y1, str(i1), bbox=dict(facecolor='r', alpha=alpha), ha='center', va='center')
            ax.text(x2, y2, str(i2), bbox=dict(facecolor='g', alpha=alpha), ha='center', va='center')
            if max_score > 0:
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth * score / max_score)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.canvas.set_window_title('limb%d (%d-%d)' % (index, i1, i2))
        fig.tight_layout()
        plt.show()


def show_clusters(image, limbs_index, limbs, parts, threshold, limits, steps, min_score, min_count, linewidth=10):
    height, width, _ = image.shape
    _height, _width, _ = limbs.shape
    scale_y, scale_x = height / _height, width / _width
    peaks = pyopenpose.featuremap_peaks(parts, threshold, limits)
    assert len(peaks) == parts.shape[-1]
    clusters = pyopenpose.clustering(limbs_index, limbs, peaks, steps, min_score, min_count)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image)
    colors = [prop['color'] for _, prop in zip(clusters, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    max_score = max(clusters, key=lambda item: item[1])[1] if clusters else 0
    for (points, score, count), color in zip(clusters, colors):
        assert len(points) == len(peaks)
        assert sum(1 for p in points if p >= 0) == count
        assert count > 0
        _linewidth = linewidth * score / max_score
        for i1, i2 in limbs_index:
            p1, p2 = points[i1], points[i2]
            if p1 >= 0 and p2 >= 0:
                y1, x1, _ = peaks[i1][p1]
                y2, x2, _ = peaks[i2][p2]
                y1, x1 = y1 * scale_y, x1 * scale_x
                y2, x2 = y2 * scale_y, x2 * scale_x
                ax.plot([x1, x2], [y1, y2], '.', color=color)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=_linewidth)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()
