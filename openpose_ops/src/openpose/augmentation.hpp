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

#include <cassert>
#include <random>
#include <utility>
#include <limits>
#include <algorithm>
#include <exception>
#include <gtest/gtest.h>
#include <boost/format.hpp>
#include <tensorflow/core/framework/tensor_types.h>
#include <opencv2/opencv.hpp>
#include "convert.hpp"
#include "render.hpp"

namespace openpose
{
template <typename _TReal>
cv::Rect_<_TReal> calc_keypoints_rect(typename tensorflow::TTypes<_TReal, 3>::Tensor keypoints, const typename tensorflow::TTypes<_TReal>::Tensor::Index index)
{
	typedef typename tensorflow::TTypes<_TReal>::Tensor::Index _TIndex;
	assert(keypoints.rank() == 3);
	assert(keypoints.dimension(2) == 3);
	_TReal xmin = std::numeric_limits<_TReal>::max(), xmax = std::numeric_limits<_TReal>::min();
	_TReal ymin = std::numeric_limits<_TReal>::max(), ymax = std::numeric_limits<_TReal>::min();
	for (_TIndex i = 0; i < keypoints.dimension(1); ++i)
	{
		if (keypoints(index, i, 2) > 0)
		{
			const _TReal x = keypoints(index, i, 0);
			if (x < xmin)
				xmin = x;
			if (x > xmax)
				xmax = x;
			const _TReal y = keypoints(index, i, 1);
			if (y < ymin)
				ymin = y;
			if (y > ymax)
				ymax = y;
		}
	}
	if (xmax < xmin || ymax < ymin)
		throw std::runtime_error((boost::format("%d has no point") % index).str());
	return cv::Rect_<_TReal>(xmin, ymin, xmax - xmin, ymax - ymin);
}

template <typename _TReal>
void rotate_points(const cv::Mat &rotate_mat, typename tensorflow::TTypes<_TReal, 3>::Tensor keypoints)
{
	typedef typename tensorflow::TTypes<_TReal>::Tensor::Index _TIndex;
	typedef double _TRotate;
	assert(keypoints.rank() == 3);
	assert(keypoints.dimension(2) == 3);
	for (_TIndex i = 0; i < keypoints.dimension(0); ++i)
		for (_TIndex j = 0; j < keypoints.dimension(1); ++j)
			if (keypoints(i, j, 2) > 0)
			{
				cv::Mat point(3, 1, rotate_mat.type());
				point.at<_TRotate>(0, 0) = keypoints(i, j, 0);
				point.at<_TRotate>(1, 0) = keypoints(i, j, 1);
				point.at<_TRotate>(2, 0) = 1;
				point = rotate_mat * point;
				keypoints(i, j, 0) = point.at<_TRotate>(0, 0);
				keypoints(i, j, 1) = point.at<_TRotate>(1, 0);
			}
}

template <typename _TPixel, typename _TReal>
void center_rotate(const _TReal rotate, const cv::Mat_<cv::Vec<_TPixel, 3> > &image, const cv::Mat_<_TPixel> &mask, cv::Mat_<cv::Vec<_TPixel, 3> > &image_result, cv::Mat_<_TPixel> &mask_result, typename tensorflow::TTypes<_TReal, 3>::Tensor keypoints_result)
{
	typedef double _TRotate;
	const cv::Point_<_TReal> center(image.cols / 2.0, image.rows / 2.0);
	cv::Mat rotate_mat = cv::getRotationMatrix2D(center, rotate, 1.0);
	cv::Rect rect = cv::RotatedRect(center, image.size(), rotate).boundingRect();
	rotate_mat.at<_TRotate>(0, 2) += rect.width / 2.0 - center.x;
	rotate_mat.at<_TRotate>(1, 2) += rect.height / 2.0 - center.y;
	cv::warpAffine(image, image_result, rotate_mat, rect.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar_<_TPixel>(128, 128, 128));
	cv::warpAffine(mask, mask_result, rotate_mat, rect.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar_<_TPixel>(255));
	rotate_points<_TReal>(rotate_mat, keypoints_result);
}

template <typename _TReal>
cv::Rect_<_TReal> calc_bound_size(_TReal range, const cv::Size &size, const cv::Size &dsize)
{
	if (range <= 0)
		range = std::min(size.width, size.height);
	cv::Rect_<_TReal> bound;
	if (size.width < size.height)
	{
		bound.width = std::min<_TReal>(range, size.width);
		bound.height = bound.width * dsize.height / dsize.width;
	}
	else
	{
		bound.height = std::min<_TReal>(range, size.height);
		bound.width = bound.height * dsize.width / dsize.height;
	}
	assert(bound.width <= size.width && bound.height <= size.height);
	return bound;
}

template <typename _TRandom, typename _TReal>
void update_bound_pos(_TRandom &random, const cv::Rect_<_TReal> &keypoints_rect, const cv::Size &size, cv::Rect_<_TReal> &bound)
{
	assert(bound.width <= size.width && bound.height <= size.height);
	const cv::Point_<_TReal> keypoints_br = keypoints_rect.br();
	assert(keypoints_rect.x >= 0 && keypoints_rect.y >= 0);
	assert(keypoints_br.x < size.width && keypoints_br.y < size.height);
	cv::Point_<_TReal> xy1(keypoints_br.x - bound.width, keypoints_br.y - bound.height);
	if (xy1.x <= 0)
		xy1.x = 0;
	if (xy1.y <= 0)
		xy1.y = 0;
	cv::Point_<_TReal> xy2(keypoints_rect.x, keypoints_rect.y);
	if (xy2.x + bound.width > size.width)
		xy2.x = size.width - bound.width;
	if (xy2.y + bound.height > size.height)
		xy2.y = size.height - bound.height;
	const std::pair<_TReal, _TReal> xrange = std::minmax(xy1.x, xy2.x);
	const std::pair<_TReal, _TReal> yrange = std::minmax(xy1.y, xy2.y);
	bound.x = std::uniform_real_distribution<_TReal>(xrange.first, xrange.second)(random);
	bound.y = std::uniform_real_distribution<_TReal>(yrange.first, yrange.second)(random);
}

template <typename _TReal>
void move_scale_keypoints(const cv::Rect_<_TReal> &bound, const cv::Size &dsize, typename tensorflow::TTypes<_TReal, 3>::Tensor keypoints)
{
	typedef typename tensorflow::TTypes<_TReal>::Tensor::Index _TIndex;
	assert(keypoints.rank() == 3);
	assert(keypoints.dimension(2) == 3);
	for (_TIndex i = 0; i < keypoints.dimension(0); ++i)
		for (_TIndex j = 0; j < keypoints.dimension(1); ++j)
			if (keypoints(i, j, 2) > 0)
			{
				keypoints(i, j, 0) = (keypoints(i, j, 0) - bound.x) * dsize.width / bound.width;
				keypoints(i, j, 1) = (keypoints(i, j, 1) - bound.y) * dsize.height / bound.height;
			}
}

template <typename _TPixel, typename _TReal, typename _TRandom, typename _TIndex>
void _augmentation(_TRandom &random,
	typename tensorflow::TTypes<_TPixel, 3>::ConstTensor image, typename tensorflow::TTypes<_TPixel, 3>::ConstTensor mask, typename tensorflow::TTypes<_TReal, 3>::ConstTensor keypoints,
	const _TReal scale, const _TReal rotate,
	typename tensorflow::TTypes<_TPixel, 3>::Tensor image_result, typename tensorflow::TTypes<_TPixel, 3>::Tensor mask_result, typename tensorflow::TTypes<_TReal, 3>::Tensor keypoints_result,
	const _TIndex index
)
{
	typedef cv::Vec<_TPixel, 3> _TVec3;
	typedef cv::Mat_<_TVec3> _TMat3;
	typedef cv::Mat_<_TPixel> _TMat1;

	assert(scale > 1);
	keypoints_result = keypoints;
	_TMat3 _image_result;
	_TMat1 _mask_result;
	center_rotate(rotate, tensor_mat<_TPixel, 3>(image), tensor_mat<_TPixel>(mask), _image_result, _mask_result, keypoints_result);
#if 1
#ifdef DEBUG_SHOW
	{
		const cv::Mat canvas = render<_TPixel>(_image_result, _mask_result, keypoints_result, index);
		cv::imshow("center_rotate", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	const cv::Size size(_image_result.cols, _image_result.rows);
	const cv::Size dsize(image_result.dimension(1), image_result.dimension(0));
	const cv::Rect_<_TReal> keypoints_rect = calc_keypoints_rect<_TReal>(keypoints_result, index);
	const _TReal range = std::max(keypoints_rect.width, keypoints_rect.height);
	cv::Rect_<_TReal> bound = calc_bound_size(range * scale, size, dsize);
	update_bound_pos(random, keypoints_rect, size, bound);
	_image_result = _image_result(bound);
	_mask_result = _mask_result(bound);
#if 1
#ifdef DEBUG_SHOW
	{
		const cv::Mat canvas = render<_TPixel>(_image_result, _mask_result);
		cv::imshow("crop", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	resize(_image_result, _image_result, dsize, 0, 0, cv::INTER_CUBIC);
	resize(_mask_result, _mask_result, dsize, 0, 0, cv::INTER_CUBIC);
	move_scale_keypoints(bound, dsize, keypoints_result);
#if 1
#ifdef DEBUG_SHOW
	{
		const cv::Mat canvas = render<_TPixel>(_image_result, _mask_result, keypoints_result, index);
		cv::imshow("scale", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	copy_mat_tensor<_TPixel, 3>(_image_result, image_result);
	copy_mat_tensor<_TPixel>(_mask_result, mask_result);
}

template <typename _TPixel, typename _TReal, typename _TRandom>
typename tensorflow::TTypes<_TPixel>::Tensor::Index augmentation(_TRandom &random,
	typename tensorflow::TTypes<_TPixel, 3>::ConstTensor image, typename tensorflow::TTypes<_TPixel, 3>::ConstTensor mask, typename tensorflow::TTypes<_TReal, 3>::ConstTensor keypoints,
	const _TReal scale, const _TReal rotate,
	typename tensorflow::TTypes<_TPixel, 3>::Tensor image_result, typename tensorflow::TTypes<_TPixel, 3>::Tensor mask_result, typename tensorflow::TTypes<_TReal, 3>::Tensor keypoints_result
)
{
	typedef typename tensorflow::TTypes<_TPixel>::Tensor::Index _TIndex;
	const _TIndex index = std::uniform_int_distribution<_TIndex>(0, keypoints_result.dimension(0) - 1)(random);
	_augmentation<_TPixel>(random, image, mask, keypoints, scale, rotate, image_result, mask_result, keypoints_result, index);
	return index;
}
}
