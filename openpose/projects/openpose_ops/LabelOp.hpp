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
#include <fstream>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <tensorflow/core/framework/op_kernel.h>
#include <openpose/data/label.hpp>
#ifdef ENABLE_NPY
#include <openpose/npy.hpp>
#endif

template <typename _TReal, typename _TInteger>
class LabelOp : public tensorflow::OpKernel
{
public:
	typedef _TReal TReal;
	typedef _TInteger TInteger;

	explicit LabelOp(tensorflow::OpKernelConstruction *context);
	void Compute(tensorflow::OpKernelContext *context) override;
};

template <typename _TReal, typename _TInteger>
LabelOp<_TReal, _TInteger>::LabelOp(tensorflow::OpKernelConstruction *context)
	: tensorflow::OpKernel(context)
{
}

template <typename _TReal, typename _TInteger>
void LabelOp<_TReal, _TInteger>::Compute(tensorflow::OpKernelContext *context)
{
	const auto size_image = context->input(0).vec<TInteger>();
	const auto size_feature = context->input(1).vec<TInteger>();
	const tensorflow::Tensor &keypoints = context->input(2);
	const tensorflow::Tensor &limbs_index = context->input(3);
	const TReal sigma_parts = context->input(4).scalar<TReal>()(0);
	const TReal sigma_limbs = context->input(5).scalar<TReal>()(0);

	tensorflow::Tensor *_limbs = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape({size_feature(0), size_feature(1), limbs_index.shape().dim_size(0) * 2}), &_limbs));
	tensorflow::Tensor *_parts = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape({size_feature(0), size_feature(1), keypoints.shape().dim_size(1) + 1}), &_parts));

	try
	{
		auto _keypoints = keypoints.tensor<TReal, 3>();
		openpose::data::make_limbs(_keypoints, limbs_index.tensor<TInteger, 2>(), sigma_limbs, size_image(0), size_image(1), _limbs->tensor<TReal, 3>());
		openpose::data::make_parts(_keypoints, sigma_parts, size_image(0), size_image(1), _parts->tensor<TReal, 3>());
	}
	catch (boost::exception &e)
	{
		const boost::uuids::uuid uuid = boost::uuids::random_generator()();
#ifdef NDEBUG
		const std::string prefix = boost::uuids::to_string(uuid);
#else
		const std::string prefix = CMAKE_BINARY_DIR "/" + boost::uuids::to_string(uuid);
#endif
#ifdef ENABLE_NPY
		openpose::save_npy<tensorflow::int32>(keypoints.tensor<TReal, 3>(), prefix + ".npy");
#endif
		std::ofstream fs(prefix + ".txt");
		fs << boost::diagnostic_information(e);
		fs.close();
		std::cerr << prefix << std::endl;
		throw;
	}
}
