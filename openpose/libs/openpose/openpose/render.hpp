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
#include <cmath>
#include <type_traits>
#include <boost/format.hpp>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace openpose
{
template <typename _TPixel, int cn, typename _TTensor, int Options>
void draw_keypoints(cv::Mat_<cv::Vec<_TPixel, cn> > &mat, Eigen::TensorMap<_TTensor, Options> keypoints, const Eigen::DenseIndex index = -1, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
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

template <typename _TPixel, int cn, typename _TTensor, int Options>
void draw_grid(cv::Mat_<cv::Vec<_TPixel, cn> > &mat, Eigen::TensorMap<_TTensor, Options> bbox, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename _TTensor::Scalar _T;

	assert(bbox.dimension(2) == 4);
	const _T bbox_height = (_T)mat.rows / bbox.dimension(0), bbox_width = (_T)mat.cols / bbox.dimension(1);
	for (_TIndex i = 0; i < bbox.dimension(0); ++i)
	{
		const _T y = i * bbox_height;
		cv::line(mat, cv::Point(0, y), cv::Point(mat.cols - 1, y), CV_RGB(255,255,255));
	}
	for (_TIndex j = 0; j < bbox.dimension(1); ++j)
	{
		const _T x = j * bbox_width;
		cv::line(mat, cv::Point(x, 0), cv::Point(x, mat.rows - 1), CV_RGB(255,255,255));
	}
}

template <typename _TPixel, int cn, typename _TTensor, int Options>
void draw_bbox(cv::Mat_<cv::Vec<_TPixel, cn> > &mat, Eigen::TensorMap<_TTensor, Options> bbox, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename _TTensor::Scalar _T;
	static const std::vector<cv::Scalar> colors = {
		CV_RGB(255, 0, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(255, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(127, 0, 0),
		CV_RGB(0, 127, 0),
		CV_RGB(0, 0, 127),
		CV_RGB(0, 127, 127),
		CV_RGB(127, 0, 127),
		CV_RGB(127, 127, 0)
	};

	assert(bbox.dimension(2) == 4);
	const _T bbox_height = mat.rows / bbox.dimension(0), bbox_width = mat.cols / bbox.dimension(1);
	cv::Mat_<cv::Vec<_TPixel, cn> > mat_grid = mat.clone();
	for (_TIndex i = 0; i < bbox.dimension(0); ++i)
		for (_TIndex j = 0; j < bbox.dimension(1); ++j)
		{
			const auto &color = colors[(i * bbox.dimension(1) + j) % colors.size()];
			if (bbox(i, j, 2) > 0 && bbox(i, j, 3) > 0)
			{
				const _T bbox_y = i * bbox_height, bbox_x = j * bbox_width;
				cv::rectangle(mat_grid, cv::Point(bbox_x, bbox_y), cv::Point(bbox_x + bbox_width, bbox_y + bbox_height), color, -1);
				const _T y = bbox_y + bbox(i, j, 1), x = bbox_x + bbox(i, j, 0);
				cv::rectangle(mat, cv::Point(x, y), cv::Point(x + bbox(i, j, 2), y + bbox(i, j, 3)), color);
			}
			else
				assert(bbox(i, j, 0) == 0 && bbox(i, j, 1) == 0 && bbox(i, j, 2) == 0 && bbox(i, j, 3) == 0);
		}
	cv::addWeighted(mat, 0.7, mat_grid, 0.3, 0, mat);
}

template <typename _TPixel, int cn>
cv::Mat_<cv::Vec<_TPixel, cn> > render(const cv::Mat_<cv::Vec<_TPixel, cn> > &image, const cv::Mat_<_TPixel> &mask)
{
	typedef cv::Vec<_TPixel, cn> _TVec;
	typedef cv::Mat_<_TVec> _TMat;
	_TMat canvas = image.clone();
	cv::Mat_<_TPixel> _mask;
	cv::resize(mask, _mask, cv::Size(image.cols, image.rows));
	for (int i = 0; i < canvas.rows; ++i)
		for (int j = 0; j < canvas.cols; ++j)
			if (_mask(i, j) < 128)
			{
				_TVec &pixel = canvas(i, j);
				for (int k = 0; k < pixel.channels; ++k)
					pixel[k] = 0;
			}
	return canvas;
}

template <typename _TPixel, int cn, typename _TTensor, int Options>
cv::Mat_<cv::Vec<_TPixel, cn> > render(const cv::Mat_<cv::Vec<_TPixel, cn> > &image, const cv::Mat_<_TPixel> &mask, Eigen::TensorMap<_TTensor, Options> keypoints, const Eigen::DenseIndex index = -1)
{
	auto canvas = render(image, mask);
	draw_keypoints(canvas, keypoints, index);
	return canvas;
}

template <typename _TPixel, int cn, typename _TTensor, int Options>
cv::Mat_<cv::Vec<_TPixel, cn> > render(const cv::Mat_<cv::Vec<_TPixel, cn> > &image, const cv::Mat_<_TPixel> &mask, Eigen::TensorMap<_TTensor, Options> keypoints, Eigen::TensorMap<_TTensor, Options> bbox)
{
	auto canvas = render(image, mask);
	draw_keypoints(canvas, keypoints);
	draw_grid(canvas, bbox);
	draw_bbox(canvas, bbox);
	return canvas;
}

template <typename _TPixel, int cn, typename _TTensor, int Options>
cv::Mat_<cv::Vec<_TPixel, cn> > render(const cv::Mat_<cv::Vec<_TPixel, cn> > &image, Eigen::TensorMap<_TTensor, Options> label, const Eigen::DenseIndex index)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename std::remove_const<typename _TTensor::Scalar>::type _T;
	typedef cv::Vec<_TPixel, cn> _TVec;
	typedef cv::Mat_<_TVec> _TMat;

	cv::Mat mat(label.dimension(0), label.dimension(1), CV_8UC1);
	for (_TIndex i = 0; i < mat.rows; ++i)
		for (_TIndex j = 0; j < mat.cols; ++j)
		{
			assert(0 <= label(i, j, index) <= 1);
			mat.at<uchar>(i, j) = label(i, j, index) * 255;
		}
	cv::resize(mat, mat, cv::Size(image.cols, image.rows));
	applyColorMap(mat, mat, cv::COLORMAP_JET);
	_TMat canvas;
	cv::addWeighted(image, 0.5, mat, 0.5, 0.0, canvas);
	return canvas;
}

template <typename _TPixel, int cn, typename _T, typename _TTensor, int Options>
cv::Mat_<cv::Vec<_TPixel, cn> > render(const cv::Mat_<cv::Vec<_TPixel, cn> > &image, const std::pair<Eigen::DenseIndex, Eigen::DenseIndex> &limb_index, Eigen::TensorMap<_TTensor, Options> parts, const std::vector<std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > > &peaks, const std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &connections, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef cv::Vec<_TPixel, cn> _TVec;
	typedef cv::Mat_<_TVec> _TMat;

	assert(image.rows > 0 && image.cols > 0);
	assert(parts.dimension(0) > 0 && parts.dimension(1) > 0);

	_TMat canvas = image.clone();
	for (auto c = connections.begin(); c != connections.end(); ++c)
	{
		const auto &connection = *c;
		const auto &_p1 = peaks[limb_index.first][std::get<0>(connection)];
		const _T y1 = std::get<0>(_p1) * image.rows / parts.dimension(0), x1 = std::get<1>(_p1) * image.cols / parts.dimension(1);
		const auto &_p2 = peaks[limb_index.second][std::get<1>(connection)];
		const _T y2 = std::get<0>(_p2) * image.rows / parts.dimension(0), x2 = std::get<1>(_p2) * image.cols / parts.dimension(1);
		cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), CV_RGB(0, 0, 0), 3);
		cv::putText(canvas, (boost::format("%d") % limb_index.first).str(), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1, 20);
		cv::putText(canvas, (boost::format("%d") % limb_index.second).str(), cv::Point(x2, y2), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1, 20);
	}
	return canvas;
}

template <typename _TPixel, int cn, typename _T, typename _TTensor, int Options>
cv::Mat_<cv::Vec<_TPixel, cn> > render(const cv::Mat_<cv::Vec<_TPixel, cn> > &image, const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > &limbs_index, Eigen::TensorMap<_TTensor, Options> parts, const std::vector<std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > > &peaks, const std::list<std::tuple<std::vector<Eigen::DenseIndex>, _T, Eigen::DenseIndex> > &clusters, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef cv::Vec<_TPixel, cn> _TVec;
	typedef cv::Mat_<_TVec> _TMat;
	static const std::vector<cv::Scalar> colors = {
		CV_RGB(255, 0, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(255, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(127, 0, 0),
		CV_RGB(0, 127, 0),
		CV_RGB(0, 0, 127),
		CV_RGB(0, 127, 127),
		CV_RGB(127, 0, 127),
		CV_RGB(127, 127, 0)
	};

	assert(image.rows > 0 && image.cols > 0);
	assert(parts.dimension(0) > 0 && parts.dimension(1) > 0);

	_TMat canvas = image.clone();
	size_t index = 0;
	for (auto c = clusters.begin(); c != clusters.end(); ++c)
	{
		const auto &cluster = *c;
		const std::vector<_TIndex> &points = std::get<0>(cluster);
		assert(points.size() == parts.dimension(2));
		const auto &color = colors[index % colors.size()];
		for (size_t l = 0; l < limbs_index.size(); ++l)
		{
			const std::pair<_TIndex, _TIndex> &limb_index = limbs_index[l];
			const _TIndex p1 = points[limb_index.first], p2 = points[limb_index.second];
			if (p1 >= 0 && p2 >= 0)
			{
				const auto &_p1 = peaks[limb_index.first][p1];
				const _T y1 = std::get<0>(_p1) * image.rows / parts.dimension(0), x1 = std::get<1>(_p1) * image.cols / parts.dimension(1);
				const auto &_p2 = peaks[limb_index.second][p2];
				const _T y2 = std::get<0>(_p2) * image.rows / parts.dimension(0), x2 = std::get<1>(_p2) * image.cols / parts.dimension(1);
				cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
				cv::putText(canvas, (boost::format("%d") % limb_index.first).str(), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 20);
				cv::putText(canvas, (boost::format("%d") % limb_index.second).str(), cv::Point(x2, y2), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 20);
			}
		}
		index += 1;
	}
	return canvas;
}
}
