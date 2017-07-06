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
template <typename _TTensor, int Options>
void init_bbox(Eigen::TensorMap<_TTensor, Options> xy_offset, Eigen::TensorMap<_TTensor, Options> width_height, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;

	ASSERT_OPENPOSE(xy_offset.dimension(0) == width_height.dimension(0) && xy_offset.dimension(1) == width_height.dimension(1));
	ASSERT_OPENPOSE(xy_offset.dimension(2) == 2);
	ASSERT_OPENPOSE(width_height.dimension(2) == 2);
	for (_TIndex i = 0; i < width_height.dimension(0); ++i)
		for (_TIndex j = 0; j < width_height.dimension(1); ++j)
		{
			for (_TIndex k = 0; k < xy_offset.dimension(2); ++k)
				xy_offset(i, j, k) = 0;
			for (_TIndex k = 0; k < width_height.dimension(2); ++k)
				width_height(i, j, k) = 0;
		}
}

template <typename _TConstTensor, typename _TTensor, int Options>
size_t update_bbox(const cv::Size &dsize, const typename _TTensor::Scalar scale, Eigen::TensorMap<_TConstTensor, Options> keypoints, Eigen::TensorMap<_TTensor, Options> xy_offset, Eigen::TensorMap<_TTensor, Options> width_height, typename std::enable_if<_TConstTensor::NumIndices == 3>::type* = nullptr, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename _TTensor::Scalar _T;

	ASSERT_OPENPOSE(xy_offset.dimension(0) == width_height.dimension(0) && xy_offset.dimension(1) == width_height.dimension(1));
	ASSERT_OPENPOSE(xy_offset.dimension(2) == 2);
	ASSERT_OPENPOSE(width_height.dimension(2) == 2);
	const _T bbox_height = (_T)dsize.height / width_height.dimension(0), bbox_width = (_T)dsize.width / width_height.dimension(1);
	size_t count = 0;
	for (_TIndex index = 0; index < keypoints.dimension(0); ++index)
	{
		const cv::Rect_<_T> rect = calc_keypoints_rect(keypoints, index, dsize);
		if (rect.width > 0 && rect.height > 0)
		{
			const _T _width2 = rect.width * scale / 2, _height2 = rect.height * scale / 2;
			const _T _x = rect.x + rect.width / 2, _y = rect.y + rect.height / 2;
			const _T xmin = std::max<_T>(_x - _width2, 0), xmax = std::min<_T>(_x + _width2, dsize.width - 1);
			const _T ymin = std::max<_T>(_y - _height2, 0), ymax = std::min<_T>(_y + _height2, dsize.height - 1);
			const _T x = (xmin + xmax) / 2, y = (ymin + ymax) / 2;
			const _TIndex ix = x / bbox_width, iy = y / bbox_height;
			if (0 <= iy && iy < xy_offset.dimension(0) && 0 <= ix && ix < xy_offset.dimension(1))
			{
				const _T x_offset = x - ix * bbox_width, y_offset = y - iy * bbox_height;
				ASSERT_OPENPOSE(x_offset >= 0 && y_offset >= 0);
				xy_offset(iy, ix, 0) = x_offset;
				xy_offset(iy, ix, 1) = y_offset;
				width_height(iy, ix, 0) = xmax - xmin;
				width_height(iy, ix, 1) = ymax - ymin;
				++count;
			}
		}
	}
	ASSERT_OPENPOSE(count > 0);
	return count;
}

template <typename _TConstTensor, typename _TTensor, int Options>
size_t keypoints_bbox(const Eigen::DenseIndex height, const Eigen::DenseIndex width, const typename _TTensor::Scalar scale, Eigen::TensorMap<_TConstTensor, Options> keypoints, Eigen::TensorMap<_TTensor, Options> xy_offset, Eigen::TensorMap<_TTensor, Options> width_height)
{
	typedef Eigen::DenseIndex _TIndex;

	ASSERT_OPENPOSE(scale >= 1);
	init_bbox(xy_offset, width_height);
	const cv::Size dsize(width, height);
	return update_bbox(dsize, scale, keypoints, xy_offset, width_height);
}
}
}
