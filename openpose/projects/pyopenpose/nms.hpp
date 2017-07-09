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
#include <openpose/postprocess/nms.hpp>
#include "convert.hpp"

template <typename _T>
std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > feature_peaks(pybind11::array_t<_T> feature, const _T threshold, const _T radius)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_T, 2, Eigen::RowMajor, _TIndex> _TTensor;
	const auto _feature = numpy_tensor<_TTensor>(feature);
	auto peaks = openpose::postprocess::feature_peaks(typename tensorflow::TTypes<_T, 2>::ConstTensor(_feature.data(), _feature.dimensions()), threshold);
	return openpose::postprocess::filter_peaks(peaks, radius);
}

template <typename _T>
std::vector<std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > > featuremap_peaks(pybind11::array_t<_T> featuremap, const _T threshold, const _T radius)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	const auto _featuremap = numpy_tensor<_TTensor>(featuremap);
	return openpose::postprocess::featuremap_peaks(typename tensorflow::TTypes<_T, 3>::ConstTensor(_featuremap.data(), _featuremap.dimensions()), threshold, radius);
}
