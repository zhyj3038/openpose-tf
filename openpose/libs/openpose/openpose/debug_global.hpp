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

#include <tensorflow/core/framework/tensor_types.h>
#include <opencv2/opencv.hpp>

namespace openpose
{
extern cv::Mat_<cv::Vec<uchar, 3> > image_;
extern Eigen::Tensor<float, 3, Eigen::RowMajor, Eigen::DenseIndex> parts_;
}
