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
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>

namespace openpose
{
template <typename _TTensor>
void draw_keypoints(cv::Mat &mat, const _TTensor &keypoints, typename _TTensor::Index index)
{
	assert(keypoints.rank() == 3);
	for (typename _TTensor::Index i = 0; i < keypoints.dimension(0); ++i)
	{
		for (typename _TTensor::Index j = 0; j < keypoints.dimension(1); ++j)
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
	cv::Mat canvas;
	cv::Mat _mask_result_3c;
	cv::cvtColor(mask, _mask_result_3c, CV_GRAY2BGR);
	cv::addWeighted(image, 0.5, _mask_result_3c, 0.5, 0.0, canvas);
	return canvas;
}

template <typename _TPixel, typename _TTensor>
cv::Mat render(const cv::Mat_<cv::Vec<_TPixel, 3> > &image, const cv::Mat_<_TPixel> &mask, const _TTensor &keypoints, typename _TTensor::Index index = -1)
{
	cv::Mat canvas = render<_TPixel>(image, mask);
	draw_keypoints(canvas, keypoints, index);
	return canvas;
}
}
