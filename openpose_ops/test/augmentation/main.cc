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

template <typename _TTensor>
_TTensor mat_tensor(const cv::Mat &mat)
{
	typedef typename _TTensor::Scalar _T;
	_TTensor tensor(mat.rows, mat.cols, mat.channels());
	for (size_t i = 0; i < tensor.size(); ++i)
		tensor(i) = mat.data[i];
	return tensor;
}

template <typename _TTensor>
cv::Mat tensor_mat3(const _TTensor &tensor)
{
	assert(tensor.rank() == 3);
	assert(tensor.dimension(2) == 3);
	cv::Mat mat(tensor.dimension(0), tensor.dimension(1), CV_8UC3);
	for (typename _TTensor::Index i = 0; i < tensor.dimension(0); ++i)
		for (typename _TTensor::Index j = 0; j < tensor.dimension(1); ++j)
		{
			cv::Vec3b &pixel = mat.at<cv::Vec3b>(i, j);
			for (typename _TTensor::Index k = 0; k < tensor.dimension(2); ++k)
				pixel.val[k] = tensor(i, j, k);
		}
	return mat;
}

template <typename _TTensor>
cv::Mat1b tensor_mat1(const _TTensor &tensor)
{
	assert(tensor.rank() == 3);
	assert(tensor.dimension(2) == 1);
	cv::Mat1b mat(tensor.dimension(0), tensor.dimension(1));
	for (typename _TTensor::Index i = 0; i < tensor.dimension(0); ++i)
		for (typename _TTensor::Index j = 0; j < tensor.dimension(1); ++j)
			mat.at<uchar>(i, j) = tensor(i, j, 0);
	return mat;
}

template <typename _TPixel, typename _TReal, typename _TRandom>
void test(const std::string &path_image, const std::string &path_mask, const std::string &path_keypoints, const size_t height, const size_t width, const _TReal scale, const _TReal rotate, _TRandom &random, int index = -1)
{
	typedef Eigen::Tensor<_TPixel, 3, Eigen::RowMajor, Eigen::DenseIndex> _TTensorPixel;
	typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, Eigen::DenseIndex> _TTensorReal;

	const cv::Mat image = cv::imread(path_image, CV_LOAD_IMAGE_COLOR);
	const cv::Mat1b mask = cv::imread(path_mask, cv::IMREAD_GRAYSCALE);

	const _TTensorPixel _image = mat_tensor<_TTensorPixel>(image);
	const _TTensorPixel _mask = mat_tensor<_TTensorPixel>(mask);
	const _TTensorReal _keypoints = openpose::load_npy3<tensorflow::int32, _TTensorReal>(path_keypoints);
	_TTensorPixel _image_result(height, width, _image.dimension(2));
	_TTensorPixel _mask_result(height, width, _mask.dimension(2));
	_TTensorReal _keypoints_result(_keypoints.dimensions());
	if (index != -1)
	{
		assert(0 <= index && index < _keypoints.dimension(0));
		openpose::_augmentation<_TPixel, _TReal>(random,
			typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_image.data(), _image.dimensions()),
			typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_mask.data(), _mask.dimensions()),
			typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_keypoints.data(), _keypoints.dimensions()),
			scale, rotate,
			typename tensorflow::TTypes<_TPixel, 3>::Tensor(_image_result.data(), _image_result.dimensions()),
			typename tensorflow::TTypes<_TPixel, 3>::Tensor(_mask_result.data(), _mask_result.dimensions()),
			typename tensorflow::TTypes<_TReal, 3>::Tensor(_keypoints_result.data(), _keypoints_result.dimensions()),
			index
		);
	}
	else
		index = openpose::augmentation<_TPixel, _TReal>(random,
			typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_image.data(), _image.dimensions()),
			typename tensorflow::TTypes<_TPixel, 3>::ConstTensor(_mask.data(), _mask.dimensions()),
			typename tensorflow::TTypes<_TReal, 3>::ConstTensor(_keypoints.data(), _keypoints.dimensions()),
			scale, rotate,
			typename tensorflow::TTypes<_TPixel, 3>::Tensor(_image_result.data(), _image_result.dimensions()),
			typename tensorflow::TTypes<_TPixel, 3>::Tensor(_mask_result.data(), _mask_result.dimensions()),
			typename tensorflow::TTypes<_TReal, 3>::Tensor(_keypoints_result.data(), _keypoints_result.dimensions())
		);
	{
		const cv::Mat canvas = openpose::render<_TPixel>(image, mask, _keypoints);
		cv::imshow("original", canvas);
	}
	{
		const cv::Mat canvas = openpose::render<_TPixel>(tensor_mat3(_image_result), tensor_mat1(_mask_result), _keypoints_result, index);
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
	typedef float _TReal;
#ifdef NDEBUG
	_TRandom random(std::time(0));
#else
	_TRandom random;
#endif
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_val2014_000000000136";
		const _TReal scale = 1.4529;
		const _TReal rotate = 26.8007;
		test<tensorflow::uint8, _TReal, _TRandom>(prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 368, scale, rotate, random, 1);
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_val2014_000000000241";
		const _TReal scale = 1.99419;
		const _TReal rotate = -3.95667;
		test<tensorflow::uint8, _TReal, _TRandom>(prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 368, scale, rotate, random, 0);
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_train2014_000000000036";
		const _TReal scale = 1.38945;
		const _TReal rotate = -2.13689;
		test<tensorflow::uint8, _TReal, _TRandom>(prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 368, scale, rotate, random);
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/COCO_train2014_000000000077";
		const _TReal scale = std::uniform_real_distribution<_TReal>(1, 1.5)(random);
		const _TReal rotate = std::uniform_real_distribution<_TReal>(-40, 40)(random);
		test<tensorflow::uint8, _TReal, _TRandom>(prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			184, 368, scale, rotate, random);
		test<tensorflow::uint8, _TReal, _TRandom>(prefix + IMAGE_EXT, prefix + MASK_SUFFIX, prefix + ".npy",
			368, 184, 1000, rotate, random);
	}
	return 0;
}
