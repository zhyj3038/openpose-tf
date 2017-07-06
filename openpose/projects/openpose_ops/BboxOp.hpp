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

#include <tensorflow/core/framework/op_kernel.h>
#include <openpose/data/bbox.hpp>

template <typename _TReal, typename _TInteger>
class BboxOp : public tensorflow::OpKernel
{
public:
	typedef _TReal TReal;
	typedef _TInteger TInteger;

	explicit BboxOp(tensorflow::OpKernelConstruction *context);
	void Compute(tensorflow::OpKernelContext *context) override;
};

template <typename _TReal, typename _TInteger>
BboxOp<_TReal, _TInteger>::BboxOp(tensorflow::OpKernelConstruction *context)
	: tensorflow::OpKernel(context)
{
}

template <typename _TReal, typename _TInteger>
void BboxOp<_TReal, _TInteger>::Compute(tensorflow::OpKernelContext *context)
{
	const tensorflow::Tensor &keypoints = context->input(0);
	const auto size_image = context->input(1).vec<TInteger>();
	const auto size_bbox = context->input(2).vec<TInteger>();
	const TReal scale = context->input(3).scalar<TReal>()(0);

	tensorflow::Tensor *xy_offset = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape({size_bbox(0), size_bbox(1), 2}), &xy_offset));
	tensorflow::Tensor *width_height = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape({size_bbox(0), size_bbox(1), 2}), &width_height));

	openpose::data::keypoints_bbox(size_image(0), size_image(1), scale, keypoints.tensor<TReal, 3>(), xy_offset->tensor<TReal, 3>(), width_height->tensor<TReal, 3>());
}
