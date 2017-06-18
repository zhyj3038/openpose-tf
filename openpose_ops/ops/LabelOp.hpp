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
#include <tensorflow/core/framework/op_kernel.h>
#include <openpose/label.hpp>
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
	const auto size_label = context->input(1).vec<TInteger>();
	const tensorflow::Tensor &keypoints = context->input(2);
	const tensorflow::Tensor &limbs = context->input(3);
	const TReal sigma_parts = context->input(4).scalar<TReal>()(0);
	const TReal sigma_limbs = context->input(5).scalar<TReal>()(0);

	tensorflow::Tensor *label = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape({size_label(0), size_label(1), limbs.shape().dim_size(0) * 2 + keypoints.shape().dim_size(1) + 1}), &label));

	try
	{
		openpose::make_label(
			keypoints.tensor<TReal, 3>(), limbs.tensor<TInteger, 2>(),
			sigma_limbs, sigma_parts, size_image(0), size_image(1),
			label->tensor<TReal, 3>());
	}
	catch (...)
	{
#ifdef ENABLE_NPY
		openpose::save_npy<tensorflow::int32>(keypoints.tensor<TReal, 3>(), CMAKE_BINARY_DIR "/keypoints.npy");
#endif
		std::cerr << CMAKE_BINARY_DIR << std::endl;
		throw;
	}
}
