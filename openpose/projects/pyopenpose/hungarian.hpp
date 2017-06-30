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
#include <openpose/postprocess/hungarian.hpp>
#include "convert.hpp"

template <typename _T>
std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > calc_limb_score(const Eigen::DenseIndex channel, pybind11::array_t<_T> limbs, const std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &peaks1, const std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &peaks2, const size_t steps, const _T min_score, const size_t min_count)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	const auto _limbs = numpy_tensor<_TTensor>(limbs);
	return openpose::postprocess::calc_limb_score(channel, typename tensorflow::TTypes<_T, 3>::ConstTensor(_limbs.data(), _limbs.dimensions()), peaks1, peaks2, steps, min_score, min_count);
}

template <typename _T>
std::list<std::tuple<std::vector<Eigen::DenseIndex>, _T, Eigen::DenseIndex> > clustering(const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > &limbs_index, pybind11::array_t<_T> limbs, const std::vector<std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > > &peaks, const size_t steps, const _T min_score, const size_t min_count)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	const auto _limbs = numpy_tensor<_TTensor>(limbs);
	return openpose::postprocess::clustering(limbs_index, typename tensorflow::TTypes<_T, 3>::ConstTensor(_limbs.data(), _limbs.dimensions()), peaks, steps, min_score, min_count);
}
