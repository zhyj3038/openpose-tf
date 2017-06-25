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

namespace openpose
{
namespace data
{
template <typename _TTensor, int Options>
cv::Rect_<typename _TTensor::Scalar> calc_keypoints_rect(Eigen::TensorMap<_TTensor, Options> keypoints, const Eigen::DenseIndex index, const cv::Size &size)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename _TTensor::Scalar _TReal;
	ASSERT_OPENPOSE(keypoints.rank() == 3);
	ASSERT_OPENPOSE(keypoints.dimension(2) == 3);
	_TReal xmin = std::numeric_limits<_TReal>::max(), xmax = std::numeric_limits<_TReal>::min();
	_TReal ymin = std::numeric_limits<_TReal>::max(), ymax = std::numeric_limits<_TReal>::min();
	for (_TIndex i = 0; i < keypoints.dimension(1); ++i)
	{
		if (keypoints(index, i, 2) > 0)
		{
			const _TReal x = std::min<_TReal>(std::max<_TReal>(keypoints(index, i, 0), 0), size.width - 1);
			if (x < xmin)
				xmin = x;
			if (x > xmax)
				xmax = x;
			const _TReal y = std::min<_TReal>(std::max<_TReal>(keypoints(index, i, 1), 0), size.height - 1);
			if (y < ymin)
				ymin = y;
			if (y > ymax)
				ymax = y;
		}
	}
	ASSERT_OPENPOSE(xmin <= xmax && ymin <= ymax);
	return cv::Rect_<_TReal>(xmin, ymin, xmax - xmin, ymax - ymin);
}

template <typename _TTensor, int Options>
void rotate_points(const cv::Mat &rotate_mat, Eigen::TensorMap<_TTensor, Options> keypoints)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef double _TRotate;
	ASSERT_OPENPOSE(keypoints.rank() == 3);
	ASSERT_OPENPOSE(keypoints.dimension(2) == 3);
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

template <typename _TPixel, typename _TTensor, int Options>
void center_rotate(const typename _TTensor::Scalar rotate, const cv::Mat_<cv::Vec<_TPixel, 3> > &image, const cv::Mat_<_TPixel> &mask, cv::Mat_<cv::Vec<_TPixel, 3> > &image_result, cv::Mat_<_TPixel> &mask_result, Eigen::TensorMap<_TTensor, Options> keypoints_result, const _TPixel fill)
{
	typedef typename _TTensor::Scalar _TReal;
	typedef double _TRotate;

	const cv::Point_<_TReal> center(image.cols / 2.0, image.rows / 2.0);
	cv::Mat rotate_mat = cv::getRotationMatrix2D(center, rotate, 1.0);
	cv::Rect rect = cv::RotatedRect(center, image.size(), rotate).boundingRect();
	rotate_mat.at<_TRotate>(0, 2) += rect.width / 2.0 - center.x;
	rotate_mat.at<_TRotate>(1, 2) += rect.height / 2.0 - center.y;
	cv::warpAffine(image, image_result, rotate_mat, rect.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar_<_TPixel>(128, 128, 128));
	cv::warpAffine(mask, mask_result, rotate_mat, rect.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar_<_TPixel>(fill));
	rotate_points(rotate_mat, keypoints_result);
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
	ASSERT_OPENPOSE(bound.width <= size.width && bound.height <= size.height);
	return bound;
}

template <typename _TRandom, typename _TReal>
void update_bound_pos(_TRandom &random, const cv::Rect_<_TReal> &keypoints_rect, const cv::Size &size, cv::Rect_<_TReal> &bound)
{
	ASSERT_OPENPOSE(bound.width <= size.width && bound.height <= size.height);
	const cv::Point_<_TReal> keypoints_br = keypoints_rect.br();
	ASSERT_OPENPOSE(keypoints_rect.x >= 0 && keypoints_rect.y >= 0);
	ASSERT_OPENPOSE(keypoints_br.x < size.width && keypoints_br.y < size.height);
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

template <typename _TTensor, int Options>
void move_scale_keypoints(const cv::Rect_<typename _TTensor::Scalar> &bound, const cv::Size &dsize, Eigen::TensorMap<_TTensor, Options> keypoints)
{
	typedef Eigen::DenseIndex _TIndex;
	ASSERT_OPENPOSE(keypoints.rank() == 3);
	ASSERT_OPENPOSE(keypoints.dimension(2) == 3);
	for (_TIndex i = 0; i < keypoints.dimension(0); ++i)
		for (_TIndex j = 0; j < keypoints.dimension(1); ++j)
			if (keypoints(i, j, 2) > 0)
			{
				keypoints(i, j, 0) = (keypoints(i, j, 0) - bound.x) * dsize.width / bound.width;
				keypoints(i, j, 1) = (keypoints(i, j, 1) - bound.y) * dsize.height / bound.height;
			}
}

template <typename _TRandom, typename _TConstTensorPixel, typename _TConstTensorReal, typename _TTensorPixel, typename _TTensorReal, int Options>
void augmentation(_TRandom &random,
	Eigen::TensorMap<_TConstTensorPixel, Options> image, Eigen::TensorMap<_TConstTensorPixel, Options> mask, Eigen::TensorMap<_TConstTensorReal, Options> keypoints,
	const typename _TTensorReal::Scalar scale, const typename _TTensorReal::Scalar rotate,
	Eigen::TensorMap<_TTensorPixel, Options> image_result, Eigen::TensorMap<_TTensorPixel, Options> mask_result, Eigen::TensorMap<_TTensorReal, Options> keypoints_result,
	const typename _TConstTensorPixel::Scalar fill, const Eigen::DenseIndex index)
{
	typedef typename _TTensorPixel::Scalar _TPixel;
	typedef typename _TTensorReal::Scalar _TReal;
	typedef cv::Vec<_TPixel, 3> _TVec3;
	typedef cv::Mat_<_TVec3> _TMat3;
	typedef cv::Mat_<_TPixel> _TMat1;

	ASSERT_OPENPOSE(scale > 1);
	keypoints_result = keypoints;
	_TMat3 _image_result;
	_TMat1 _mask_result;
	center_rotate(rotate, tensor_mat<_TPixel, 3>(image), tensor_mat<_TPixel>(mask), _image_result, _mask_result, keypoints_result, fill);
#if 1
#ifdef DEBUG_SHOW
	{
		const cv::Mat canvas = render(_image_result, _mask_result, typename tensorflow::TTypes<_TReal, 3>::ConstTensor(keypoints_result.data(), keypoints_result.dimensions()), index);
		cv::imshow("center_rotate", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	const cv::Size size(_image_result.cols, _image_result.rows);
	const cv::Size dsize(image_result.dimension(1), image_result.dimension(0));
	const cv::Rect_<_TReal> keypoints_rect = calc_keypoints_rect(keypoints_result, index, size);
	const _TReal range = std::max(keypoints_rect.width, keypoints_rect.height);
	cv::Rect_<_TReal> bound = calc_bound_size(range * scale, size, dsize);
	update_bound_pos(random, keypoints_rect, size, bound);
	_image_result = _image_result(bound);
	_mask_result = _mask_result(bound);
#if 1
#ifdef DEBUG_SHOW
	{
		const cv::Mat canvas = render(_image_result, _mask_result);
		cv::imshow("crop", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	resize(_image_result, _image_result, dsize, 0, 0, cv::INTER_CUBIC);
	resize(_mask_result, _mask_result, cv::Size(mask_result.dimension(1), mask_result.dimension(0)), 0, 0, cv::INTER_CUBIC);
	move_scale_keypoints(bound, dsize, keypoints_result);
#if 1
#ifdef DEBUG_SHOW
	{
		const cv::Mat canvas = render(_image_result, _mask_result, typename tensorflow::TTypes<_TReal, 3>::ConstTensor(keypoints_result.data(), keypoints_result.dimensions()), index);
		cv::imshow("scale", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	copy_mat_tensor<_TPixel, 3>(_image_result, image_result);
	copy_mat_tensor<_TPixel>(_mask_result, mask_result);
}

template <typename _TRandom, typename _TConstTensorPixel, typename _TConstTensorReal, typename _TTensorPixel, typename _TTensorReal, int Options>
Eigen::DenseIndex augmentation(_TRandom &random,
	Eigen::TensorMap<_TConstTensorPixel, Options> image, Eigen::TensorMap<_TConstTensorPixel, Options> mask, Eigen::TensorMap<_TConstTensorReal, Options> keypoints,
	const typename _TTensorReal::Scalar scale, const typename _TTensorReal::Scalar rotate,
	Eigen::TensorMap<_TTensorPixel, Options> image_result, Eigen::TensorMap<_TTensorPixel, Options> mask_result, Eigen::TensorMap<_TTensorReal, Options> keypoints_result,
	const typename _TConstTensorPixel::Scalar fill)
{
	typedef Eigen::DenseIndex _TIndex;
	const _TIndex index = std::uniform_int_distribution<_TIndex>(0, keypoints_result.dimension(0) - 1)(random);
	augmentation(random, image, mask, keypoints, scale, rotate, image_result, mask_result, keypoints_result, fill, index);
	return index;
}
}
}
