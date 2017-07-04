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
#include "shape.h"
#include "BboxOp.hpp"

REGISTER_OP("Bbox")
	.Attr("TReal: {float}")
	.Attr("TInteger: {int32}")
	.Input("keypoints: TReal")
	.Input("size_image: TInteger")
	.Input("size_bbox: TInteger")
	.Input("scale: TReal")
	.Output("bbox: TReal")
	.SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
		set_shape(c, 0, 2, {4});
		return tensorflow::Status::OK();
	})
;

REGISTER_KERNEL_BUILDER(
	Name("Bbox")
		.Device(tensorflow::DEVICE_CPU)
		.TypeConstraint<float>("TReal")
		.TypeConstraint<tensorflow::int32>("TInteger")
	, BboxOp<float, tensorflow::int32>
);
