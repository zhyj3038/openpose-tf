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

#include <iostream>
#include <tensorflow/core/framework/tensor_types.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <openpose/postprocess/estimate.hpp>
#include "convert.hpp"

template <typename _T>
std::list<std::list<std::pair<std::pair<_T, _T>, std::pair<_T, _T> > > > estimate(const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > &limbs_index, pybind11::array_t<_T> limbs, pybind11::array_t<_T> parts, const _T threshold, const size_t limits, const size_t steps, const _T min_score, const size_t min_count, const _T cluster_min_score, const size_t cluster_min_count)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	const auto _limbs = numpy_tensor<_TTensor>(limbs);
	const auto _parts = numpy_tensor<_TTensor>(parts);
	return openpose::postprocess::estimate(limbs_index, typename tensorflow::TTypes<_T, 3>::ConstTensor(_limbs.data(), _limbs.dimensions()), typename tensorflow::TTypes<_T, 3>::ConstTensor(_parts.data(), _parts.dimensions()), threshold, limits, steps, min_score, min_count, cluster_min_score, cluster_min_count);
}
