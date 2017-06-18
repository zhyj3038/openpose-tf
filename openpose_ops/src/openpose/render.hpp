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
#include <tuple>
#include <type_traits>
#include <boost/format.hpp>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace openpose
{
template <typename _TTensor, int Options>
void draw_keypoints(cv::Mat &mat, Eigen::TensorMap<_TTensor, Options> keypoints, const Eigen::DenseIndex index)
{
	typedef Eigen::DenseIndex _TIndex;
	assert(keypoints.rank() == 3);
	for (_TIndex i = 0; i < keypoints.dimension(0); ++i)
	{
		for (_TIndex j = 0; j < keypoints.dimension(1); ++j)
			if (keypoints(i, j, 2) > 0)
			{
				const cv::Point org(keypoints(i, j, 0), keypoints(i, j, 1));
				const std::string text = (boost::format("%d") % j).str();
				cv::putText(mat, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1, 20);
				const auto color = i == index ? CV_RGB(255, 0, 0) : CV_RGB(0, 0, 255);
				cv::circle(mat, org, 3, color, -1);
			}
	}
}

template <typename _TPixel>
cv::Mat render(const cv::Mat_<cv::Vec<_TPixel, 3> > &image, const cv::Mat_<_TPixel> &mask)
{
	typedef cv::Vec<_TPixel, 3> _TVec3;
	typedef cv::Mat_<_TVec3> _TMat3;
	_TMat3 _mask;
	cv::cvtColor(mask, _mask, CV_GRAY2BGR);
	cv::Mat canvas;
	cv::addWeighted(image, 0.5, _mask, 0.5, 0.0, canvas);
	return canvas;
}

template <typename _TPixel, typename _TTensor, int Options>
cv::Mat render(const cv::Mat_<cv::Vec<_TPixel, 3> > &image, const cv::Mat_<_TPixel> &mask, Eigen::TensorMap<_TTensor, Options> keypoints, const Eigen::DenseIndex index = -1)
{
	cv::Mat canvas = render(image, mask);
	draw_keypoints(canvas, keypoints, index);
	return canvas;
}

template <typename _TPixel, typename _TTensor, int Options>
cv::Mat render(const cv::Mat_<cv::Vec<_TPixel, 3> > &image, Eigen::TensorMap<_TTensor, Options> label, const Eigen::DenseIndex index)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename std::remove_const<typename _TTensor::Scalar>::type _T;
	typedef cv::Vec<_T, 3> _TVec3;
	typedef cv::Mat_<_TVec3> _TMat3;

	cv::Mat mat(label.dimension(0), label.dimension(1), CV_8UC1);
	for (_TIndex i = 0; i < mat.rows; ++i)
		for (_TIndex j = 0; j < mat.cols; ++j)
			mat.at<uchar>(i, j) = label(i, j, index) * 255;
	cv::resize(mat, mat, cv::Size(image.cols, image.rows));
	applyColorMap(mat, mat, cv::COLORMAP_JET);
	cv::Mat canvas;
	cv::addWeighted(image, 0.5, mat, 0.5, 0.0, canvas);
	return canvas;
}
}
