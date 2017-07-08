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
#include "shape.h"
#include "LabelOp.hpp"

REGISTER_OP("Label")
	.Attr("TReal: {float}")
	.Attr("TInteger: {int32}")
	.Input("size_image: TInteger")
	.Input("size_feature: TInteger")
	.Input("keypoints: TReal")
	.Input("limbs_index: TInteger")
	.Input("sigma_parts: TReal")
	.Input("sigma_limbs: TReal")
	.Output("limbs: TReal")
	.Output("parts: TReal")
	.SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
		tensorflow::shape_inference::ShapeHandle keypoints;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &keypoints));
		tensorflow::shape_inference::ShapeHandle limbs_index;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &limbs_index));
		set_shape(c, 0, 1, {c->Value(c->Dim(limbs_index, 0)) * 2});
		set_shape(c, 1, 1, {c->Value(c->Dim(keypoints, 1)) + 1});
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
