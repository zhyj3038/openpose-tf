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
#include <tensorflow/core/platform/default/integral_types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <openpose/data/label.hpp>
#include <openpose/npy.hpp>
#include <openpose/tsv.hpp>
#include <openpose/render.hpp>

std::string get_title_limbs(const Eigen::DenseIndex index, const Eigen::DenseIndex total)
{
	assert(0 <= index && index < total * 2);
	const std::string xy = index % 2 ? "y" : "x";
	return (boost::format("limb %d/%d ") % (index / 2 + 1) % total).str() + xy;

}

std::string get_title_parts(const Eigen::DenseIndex index, const Eigen::DenseIndex total)
{
	assert(0 <= index && index <= total);
	if (index < total)
		return (boost::format("part %d/%d") % (index + 1) % total).str();
	else
		return "background";
}

template <typename _TPixel, typename _TInteger, typename _TReal>
void test(const std::string &path_image, const std::string &path_keypoints, const std::string &path_limbs_index, const std::pair<size_t, size_t> downsample, const _TReal sigma_parts, const _TReal sigma_limbs)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_TInteger, 2, Eigen::RowMajor, _TIndex> _TTensorInteger;
	typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex> _TTensorReal;
	typedef cv::Vec<_TPixel, 3> _TVec3;
	typedef cv::Mat_<_TVec3> _TMat3;

	const _TMat3 image = cv::imread(path_image, CV_LOAD_IMAGE_COLOR);
	const _TTensorReal keypoints = openpose::load_npy3<tensorflow::int32, _TTensorReal>(path_keypoints);
	const _TTensorInteger limbs_index = openpose::load_tsv<_TTensorInteger>(path_limbs_index);
	_TTensorReal _limbs(image.rows / downsample.first, image.cols / downsample.second, limbs_index.dimension(0) * 2);
	_TTensorReal _parts(image.rows / downsample.first, image.cols / downsample.second, keypoints.dimension(1) + 1);
	typename tensorflow::TTypes<_TReal, 3>::ConstTensor _keypoints(keypoints.data(), keypoints.dimensions());
	openpose::data::make_limbs(_keypoints, typename tensorflow::TTypes<_TInteger, 2>::ConstTensor(limbs_index.data(), limbs_index.dimensions()),
		sigma_limbs, image.rows, image.cols,
		typename tensorflow::TTypes<_TReal, 3>::Tensor(_limbs.data(), _limbs.dimensions())
	);
	openpose::data::make_parts(_keypoints,
		sigma_parts, image.rows, image.cols,
		typename tensorflow::TTypes<_TReal, 3>::Tensor(_parts.data(), _parts.dimensions())
	);
	for (_TIndex index = 0; index < _limbs.dimension(2); ++index)
	{
		const cv::Mat canvas = openpose::render(image, typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_limbs.data(), _limbs.dimensions()), index);
		cv::imshow(get_title_limbs(index, limbs_index.dimension(0)), canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	for (_TIndex index = 0; index < _parts.dimension(2); ++index)
	{
		const cv::Mat canvas = openpose::render(image, typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_parts.data(), _parts.dimensions()), index);
		cv::imshow(get_title_parts(index, keypoints.dimension(1)), canvas);
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
	const std::pair<size_t, size_t> downsample = std::make_pair(8, 8);
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_train2014_000000000077";
		test<_TPixel, _TInteger>(prefix + IMAGE_EXT, prefix + ".npy", DUMP_DIR ".tsv", downsample, sigma_parts, sigma_limbs);
	}
	return 0;
}
