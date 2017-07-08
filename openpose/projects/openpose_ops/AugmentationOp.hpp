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

#include <ctime>
#include <random>
#include <fstream>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <tensorflow/core/framework/op_kernel.h>
#include <openpose/data/augmentation.hpp>
#ifdef ENABLE_NPY
#include <openpose/npy.hpp>
#endif

template <typename _TPixel, typename _TReal, typename _TInteger>
class AugmentationOp : public tensorflow::OpKernel
{
public:
	typedef _TPixel TPixel;
	typedef _TReal TReal;
	typedef _TInteger TInteger;
	typedef std::mt19937 TRandom;

	explicit AugmentationOp(tensorflow::OpKernelConstruction *context);
	void Compute(tensorflow::OpKernelContext *context) override;

private:
	TRandom random_;
};

template <typename _TPixel, typename _TReal, typename _TInteger>
AugmentationOp<_TPixel, _TReal, _TInteger>::AugmentationOp(tensorflow::OpKernelConstruction *context)
	: tensorflow::OpKernel(context)
#ifdef NDEBUG
	, random_(std::time(0))
#endif
{
}

template <typename _TPixel, typename _TReal, typename _TInteger>
void AugmentationOp<_TPixel, _TReal, _TInteger>::Compute(tensorflow::OpKernelContext *context)
{
	const tensorflow::Tensor &image = context->input(0);
	const tensorflow::Tensor &mask = context->input(1);
	const tensorflow::Tensor &keypoints = context->input(2);
	const auto size_image = context->input(3).vec<TInteger>();
	const auto size_feature = context->input(4).vec<TInteger>();
	const auto scale = context->input(5).vec<TReal>();
	const TReal rotate = context->input(6).scalar<TReal>()(0);
	const TPixel fill = context->input(7).scalar<TPixel>()(0);

	tensorflow::Tensor *image_result = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape({size_image(0), size_image(1), image.shape().dim_size(2)}), &image_result));
	tensorflow::Tensor *mask_result = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape({size_feature(0), size_feature(1), mask.shape().dim_size(2)}), &mask_result));
	tensorflow::Tensor *keypoints_result = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(2, keypoints.shape(), &keypoints_result));

	const TReal _scale = std::uniform_real_distribution<TReal>(scale(0), scale(1))(random_);
	const TReal _rotate = std::uniform_real_distribution<TReal>(-rotate, rotate)(random_);
	try
	{
		openpose::data::augmentation(random_,
			image.tensor<TPixel, 3>(), mask.tensor<TPixel, 3>(), keypoints.tensor<TReal, 3>(),
			_scale, _rotate,
			image_result->tensor<TPixel, 3>(), mask_result->tensor<TPixel, 3>(), keypoints_result->tensor<TReal, 3>(),
			fill
		);
	}
	catch (boost::exception &e)
	{
		cv::Mat _image;
		const boost::uuids::uuid uuid = boost::uuids::random_generator()();
#ifdef NDEBUG
		const std::string prefix = boost::uuids::to_string(uuid);
#else
		const std::string prefix = CMAKE_BINARY_DIR "/" + boost::uuids::to_string(uuid);
#endif
		cv::cvtColor(openpose::tensor_mat<_TPixel, 3>(image.tensor<TPixel, 3>()), _image, cv::COLOR_BGR2RGB);
		cv::imwrite(prefix + ".jpg", _image);
		cv::imwrite(prefix + ".mask.jpg", openpose::tensor_mat<_TPixel>(mask.tensor<TPixel, 3>()));
#ifdef ENABLE_NPY
		openpose::save_npy<tensorflow::int32>(keypoints.tensor<TReal, 3>(), prefix + ".npy");
#endif
		std::ofstream fs(prefix + ".txt");
		fs << "scale=" << _scale << std::endl;
		fs << "rotate=" << _rotate << std::endl;
		fs << boost::diagnostic_information(e);
		fs.close();
		std::cerr << prefix << std::endl;
		throw;
	}
}
