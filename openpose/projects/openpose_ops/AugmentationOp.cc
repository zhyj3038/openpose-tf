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
#include "AugmentationOp.hpp"

REGISTER_OP("Augmentation")
	.Attr("TPixel: {float, uint8}")
	.Attr("TReal: {float}")
	.Attr("TInteger: {int32}")
	.Input("image: TPixel")
	.Input("mask: TPixel")
	.Input("keypoints: TReal")
	.Input("size_image: TInteger")
	.Input("size_label: TInteger")
	.Input("scale: TReal")
	.Input("rotate: TReal")
	.Input("fill: TPixel")
	.Output("image_result: TPixel")
	.Output("mask_result: TPixel")
	.Output("keypoints_result: TReal")
	.SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
		tensorflow::shape_inference::ShapeHandle image;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &image));
		set_shape(c, 0, 3, {c->Dim(image, 2)});

		tensorflow::shape_inference::ShapeHandle mask;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &mask));
		set_shape(c, 1, 4, {c->Dim(mask, 2)});

		c->set_output(2, c->input(2));
		return tensorflow::Status::OK();
	})
;

REGISTER_KERNEL_BUILDER(
	Name("Augmentation")
		.Device(tensorflow::DEVICE_CPU)
		.TypeConstraint<tensorflow::uint8>("TPixel")
		.TypeConstraint<float>("TReal")
		.TypeConstraint<tensorflow::int32>("TInteger")
	, AugmentationOp<tensorflow::uint8, float, tensorflow::int32>
);

REGISTER_KERNEL_BUILDER(
	Name("Augmentation")
		.Device(tensorflow::DEVICE_CPU)
		.TypeConstraint<float>("TPixel")
		.TypeConstraint<float>("TReal")
		.TypeConstraint<tensorflow::int32>("TInteger")
	, AugmentationOp<float, float, tensorflow::int32>
);
