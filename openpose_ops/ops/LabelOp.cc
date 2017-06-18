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

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "LabelOp.hpp"

REGISTER_OP("Label")
	.Attr("TReal: {float}")
	.Attr("TInteger: {int32}")
	.Input("size_image: TInteger")
	.Input("size_label: TInteger")
	.Input("keypoints: TReal")
	.Input("limbs: TInteger")
	.Input("sigma_parts: TReal")
	.Input("sigma_limbs: TReal")
	.Output("label: TReal")
	.SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
		const int size_index = 1;
		tensorflow::shape_inference::ShapeHandle size;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(size_index), 1, &size));
		tensorflow::shape_inference::DimensionHandle unused;
		TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 2, &unused));

		const tensorflow::Tensor* size_tensor = c->input_tensor(size_index);
		tensorflow::shape_inference::DimensionHandle width;
		tensorflow::shape_inference::DimensionHandle height;
		if (size_tensor == nullptr)
		{
		    width = c->UnknownDim();
		    height = c->UnknownDim();
		}
		else
		{
		    if (size_tensor->dtype() != tensorflow::DT_INT32) {
		    	return tensorflow::errors::InvalidArgument(
		    		"Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
		    		"but got ", DataTypeString(size_tensor->dtype()),
					" for input #", size_index,
					" in ", c->DebugString()
				);
		    }
		    const auto vec = size_tensor->vec<tensorflow::int32>();
		    height = c->MakeDim(vec(0));
		    width = c->MakeDim(vec(1));
		}

		tensorflow::shape_inference::ShapeHandle keypoints;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &keypoints));
		tensorflow::shape_inference::ShapeHandle limbs;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &limbs));

		c->set_output(0, c->MakeShape({height, width, c->MakeDim(c->Value(c->Dim(limbs, 0)) * 2 + c->Value(c->Dim(keypoints, 1)) + 1)}));
		return tensorflow::Status::OK();
	})
;

REGISTER_KERNEL_BUILDER(
	Name("Label")
		.Device(tensorflow::DEVICE_CPU)
		.TypeConstraint<float>("TReal")
		.TypeConstraint<tensorflow::int32>("TInteger")
	, LabelOp<float, tensorflow::int32>
);
