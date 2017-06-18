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

#include <ctime>
#include <random>
#include <boost/format.hpp>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <tensorflow/core/platform/default/integral_types.h>
#include <openpose/augmentation.hpp>
#include <openpose/npy.hpp>
#include <openpose/render.hpp>

template <typename _TPixel, typename _TReal, typename _TRandom>
void test(_TRandom &random, const std::string &path_image, const std::string &path_mask, const std::string &path_keypoints, const size_t height, const size_t width, const std::pair<size_t, size_t> downsample, const _TReal scale, const _TReal rotate, const _TPixel fill, Eigen::DenseIndex index = -1)
{
	typedef Eigen::Tensor<_TPixel, 3, Eigen::RowMajor, Eigen::DenseIndex> _TTensorPixel;
	typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, Eigen::DenseIndex> _TTensorReal;
	typedef cv::Vec<_TPixel, 3> _TVec3;
	typedef cv::Mat_<_TVec3> _TMat3;
	typedef cv::Mat_<_TPixel> _TMat1;

	const _TMat3 image = cv::imread(path_image, CV_LOAD_IMAGE_COLOR);
	const _TTensorPixel _image = openpose::mat_tensor<_TTensorPixel>(image);
	const _TMat1 mask = cv::imread(path_mask, cv::IMREAD_GRAYSCALE);
	const _TTensorPixel _mask = openpose::mat_tensor<_TTensorPixel>(mask);
	const _TTensorReal _keypoints = openpose::load_npy3<tensorflow::int32, _TTensorReal>(path_keypoints);
	_TTensorPixel _image_result(height, width, _image.dimension(2));
	_TTensorPixel _mask_result(height / downsample.first, width / downsample.second, _mask.dimension(2));
	_TTensorReal _keypoints_result(_keypoints.dimensions());
	if (index != -1)
	{
		assert(0 <= index && index < _keypoints.dimension(0));
		openpose::augmentation(random,
			typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_image.data(), _image.dimensions()),
			typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_mask.data(), _mask.dimensions()),
			typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_keypoints.data(), _keypoints.dimensions()),
			scale, rotate,
			typename tensorflow::TTypes<_TPixel, 3>::Tensor(_image_result.data(), _image_result.dimensions()),
			typename tensorflow::TTypes<_TPixel, 3>::Tensor(_mask_result.data(), _mask_result.dimensions()),
			typename tensorflow::TTypes<_TReal, 3>::Tensor(_keypoints_result.data(), _keypoints_result.dimensions()),
			fill, index
		);
	}
	else
		index = openpose::augmentation(random,
			typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_image.data(), _image.dimensions()),
			typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_mask.data(), _mask.dimensions()),
			typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_keypoints.data(), _keypoints.dimensions()),
			scale, rotate,
			typename tensorflow::TTypes<_TPixel, 3>::Tensor(_image_result.data(), _image_result.dimensions()),
			typename tensorflow::TTypes<_TPixel, 3>::Tensor(_mask_result.data(), _mask_result.dimensions()),
			typename tensorflow::TTypes<_TReal, 3>::Tensor(_keypoints_result.data(), _keypoints_result.dimensions()),
			fill
		);
	{
		const cv::Mat canvas = openpose::render(image, mask,
			typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_keypoints.data(), _keypoints.dimensions()));
		cv::imshow("original", canvas);
	}
	{
		const cv::Mat canvas = openpose::render(
			openpose::tensor_mat<_TPixel, 3>(typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_image_result.data(), _image_result.dimensions())),
			openpose::tensor_mat<_TPixel>(typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_mask_result.data(), _mask_result.dimensions())),
			typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_keypoints_result.data(), _keypoints_result.dimensions()),
			index);
		cv::imshow("result", canvas);
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
}

int main(void)
{
#define IMAGE_EXT ".jpg"
#define MASK_SUFFIX ".mask.jpg"
	typedef std::mt19937 _TRandom;
	typedef tensorflow::uint8 _TPixel;
	typedef float _TReal;
#ifdef NDEBUG
	_TRandom random(std::time(0));
#else
	_TRandom random;
#endif
	const std::pair<size_t, size_t> downsample = std::make_pair(8, 8);
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_val2014_000000000136";
		const _TReal scale = 1.4529;
		const _TReal rotate = 26.8007;
		test<_TPixel>(random, prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 368, downsample, scale, rotate, 0, 1);
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_val2014_000000000241";
		const _TReal scale = 1.99419;
		const _TReal rotate = -3.95667;
		test<_TPixel>(random, prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 368, downsample, scale, rotate, 255, 0);
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_train2014_000000000036";
		const _TReal scale = 1.38945;
		const _TReal rotate = -2.13689;
		test<_TPixel>(random, prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 368, downsample, scale, rotate, 255);
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_train2014_000000000077";
		const _TReal scale = std::uniform_real_distribution<_TReal>(1, 1.5)(random);
		const _TReal rotate = std::uniform_real_distribution<_TReal>(-40, 40)(random);
		test<_TPixel>(random, prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			184, 368, downsample, scale, rotate, 255);
		test<_TPixel>(random, prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 184, downsample, (_TReal)1000, rotate, 255);
		test<_TPixel>(random, prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 368, downsample, (_TReal)1000, rotate, 0);
	}
	return 0;
}
