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

tensorflow::Status set_shape(tensorflow::shape_inference::InferenceContext *c, const int index, const int index_size, const tensorflow::shape_inference::DimensionHandle &channels)
{
	tensorflow::shape_inference::ShapeHandle size;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(index_size), 1, &size));
	tensorflow::shape_inference::DimensionHandle unused;
	TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 2, &unused));

	const tensorflow::Tensor *_size = c->input_tensor(index_size);
	tensorflow::shape_inference::DimensionHandle height;
	tensorflow::shape_inference::DimensionHandle width;
	if (_size == nullptr)
	{
		height = c->UnknownDim();
		width = c->UnknownDim();
	}
	else
	{
		if (_size->dtype() != tensorflow::DT_INT32) {
			return tensorflow::errors::InvalidArgument(
				"Bad size input type: Expected DT_INT32 "
				"but got ", DataTypeString(_size->dtype()),
				" for input #", index_size,
				" in ", c->DebugString()
			);
		}
		const auto vec = _size->vec<tensorflow::int32>();
		height = c->MakeDim(vec(0));
		width = c->MakeDim(vec(1));
	}
	c->set_output(index, c->MakeShape({height, width, channels}));
	return tensorflow::Status::OK();
}
