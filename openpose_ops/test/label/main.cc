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

#ifndef NDEBUG
#define DEBUG_SHOW
#endif

#include <boost/format.hpp>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <tensorflow/core/platform/default/integral_types.h>
#include <openpose/label.hpp>
#include <openpose/npy.hpp>
#include <openpose/tsv.hpp>
#include <openpose/render.hpp>

template <typename _TPixel, typename _TInteger, typename _TReal>
void test(const std::string &path_image, const std::string &path_keypoints, const std::string &path_limbs, const std::pair<size_t, size_t> downsample, const _TReal sigma_parts, const _TReal sigma_limbs)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_TInteger, 2, Eigen::RowMajor, _TIndex> _TTensorInteger;
	typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex> _TTensorReal;
	typedef cv::Vec<_TPixel, 3> _TVec3;
	typedef cv::Mat_<_TVec3> _TMat3;

	const _TMat3 image = cv::imread(path_image, CV_LOAD_IMAGE_COLOR);
	const _TTensorReal _keypoints = openpose::load_npy3<tensorflow::int32, _TTensorReal>(path_keypoints);
	const _TTensorInteger _limbs = openpose::load_tsv<_TTensorInteger>(path_limbs);
	_TTensorReal _label(image.rows / downsample.first, image.cols / downsample.second, _limbs.dimension(0) * 2 + _keypoints.dimension(1) + 1);
	openpose::make_label(
		typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_keypoints.data(), _keypoints.dimensions()),
		typename tensorflow::TTypes<_TInteger, 2>::ConstTensor(_limbs.data(), _limbs.dimensions()),
		sigma_limbs, sigma_parts, image.rows, image.cols,
		typename tensorflow::TTypes<_TReal, 3>::Tensor(_label.data(), _label.dimensions())
	);
	for (_TIndex index = 0; index < _label.dimension(2); ++index)
	{
		const cv::Mat canvas = openpose::render(image, typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_label.data(), _label.dimensions()), index);
		cv::imshow((boost::format("%d") % index).str(), canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}

int main(void)
{
#define IMAGE_EXT ".jpg"
	typedef tensorflow::uint8 _TPixel;
	typedef tensorflow::int32 _TInteger;
	typedef float _TReal;
	const _TReal sigma_parts = 7;
	const _TReal sigma_limbs = 7;
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_train2014_000000000077";
		test<_TPixel, _TInteger>(prefix + IMAGE_EXT, prefix + ".npy", DUMP_DIR ".tsv", std::make_pair(8, 8), sigma_parts, sigma_limbs);
	}
	return 0;
}
