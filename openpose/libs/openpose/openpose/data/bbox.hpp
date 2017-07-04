/*!
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
*/

#pragma once

#include <cmath>
#include <random>
#include <utility>
#include <limits>
#include <algorithm>
#include <exception>
#include <boost/format.hpp>
#include <boost/exception/all.hpp>
#include <tensorflow/core/framework/tensor_types.h>
#include <opencv2/opencv.hpp>
#include <openpose/assert.hpp>
#include <openpose/convert.hpp>
#include <openpose/render.hpp>
#include "augmentation.hpp"

namespace openpose
{
namespace data
{
template <typename _T>
_T _fix_size(const _T center, const _T size, const Eigen::DenseIndex bound)
{
	const _T size2 = size / 2;
	const _T min = std::max<_T>(center - size2, 0), max = std::min<_T>(center + size2, bound - 1);
	return std::min(center - min, max - center) * 2;
}

template <typename _TTensor, int Options>
void init_bbox(Eigen::TensorMap<_TTensor, Options> bbox, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	ASSERT_OPENPOSE(bbox.dimension(2) == 4);
	for (_TIndex i = 0; i < bbox.dimension(0); ++i)
		for (_TIndex j = 0; j < bbox.dimension(1); ++j)
		{
			for (_TIndex k = 0; k < bbox.dimension(2); ++k)
				bbox(i, j, k) = 0;
		}
}

template <typename _TConstTensor, typename _TTensor, int Options>
size_t update_bbox(const cv::Size &dsize, const typename _TTensor::Scalar scale, Eigen::TensorMap<_TConstTensor, Options> keypoints, Eigen::TensorMap<_TTensor, Options> bbox, typename std::enable_if<_TConstTensor::NumIndices == 3>::type* = nullptr, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename _TTensor::Scalar _T;

	ASSERT_OPENPOSE(bbox.dimension(2) == 4);
	const _T bbox_height = (_T)dsize.height / bbox.dimension(0), bbox_width = (_T)dsize.width / bbox.dimension(1);
	size_t count = 0;
	for (_TIndex index = 0; index < keypoints.dimension(0); ++index)
	{
		const cv::Rect_<_T> rect = calc_keypoints_rect(keypoints, index, dsize);
		ASSERT_OPENPOSE(0 <= rect.x && rect.x + rect.width < dsize.width);
		ASSERT_OPENPOSE(0 <= rect.y && rect.y + rect.height < dsize.height);
		const _T center_x = rect.x + rect.width / 2, center_y = rect.y + rect.height / 2;
		if (rect.width > 0 && rect.height > 0)
		{
			const _T width = _fix_size(center_x, rect.width * scale, dsize.width), height = _fix_size(center_y, rect.height * scale, dsize.height);
			const _T x = center_x - width / 2, y = center_y - height / 2;
			const _TIndex ix = x / bbox_width, iy = y / bbox_height;
			if (0 <= iy && iy < bbox.dimension(0) && 0 <= ix && ix < bbox.dimension(1))
			{
				const _T offset_x = x - ix * bbox_width, offset_y = y - iy * bbox_height;
				ASSERT_OPENPOSE(offset_x >= 0 && offset_y >= 0);
				bbox(iy, ix, 0) = offset_x;
				bbox(iy, ix, 1) = offset_y;
				bbox(iy, ix, 2) = width;
				bbox(iy, ix, 3) = height;
				++count;
			}
		}
	}
	ASSERT_OPENPOSE(count > 0);
	return count;
}

template <typename _TConstTensor, typename _TTensor, int Options>
size_t keypoints_bbox(const Eigen::DenseIndex height, const Eigen::DenseIndex width, const typename _TTensor::Scalar scale, Eigen::TensorMap<_TConstTensor, Options> keypoints, Eigen::TensorMap<_TTensor, Options> bbox)
{
	typedef Eigen::DenseIndex _TIndex;

	ASSERT_OPENPOSE(scale >= 1);
	init_bbox(bbox);
	const cv::Size dsize(width, height);
	return update_bbox(dsize, scale, keypoints, bbox);
}
}
}
