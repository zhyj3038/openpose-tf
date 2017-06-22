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

#include <cassert>
#include <type_traits>
#include <tensorflow/core/framework/tensor_types.h>
#include <opencv2/opencv.hpp>

namespace openpose
{
template <typename _T, int cn, typename _TTensor, int Options>
cv::Mat_<cv::Vec<_T, cn> > tensor_mat(Eigen::TensorMap<_TTensor, Options> tensor)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef cv::Vec<_T, cn> _TVec;
	typedef cv::Mat_<_TVec> _TMat;
	assert(tensor.dimension(2) == cn);
	_TMat mat(tensor.dimension(0), tensor.dimension(1));
	for (_TIndex i = 0; i < tensor.dimension(0); ++i)
		for (_TIndex j = 0; j < tensor.dimension(1); ++j)
		{
			_TVec &pixel = mat(i, j);
			for (_TIndex k = 0; k < tensor.dimension(2); ++k)
				pixel[k] = tensor(i, j, k);
		}
	return mat;
}

template <typename _T, typename _TTensor, int Options>
cv::Mat_<_T> tensor_mat(Eigen::TensorMap<_TTensor, Options> tensor)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef cv::Mat_<_T> _TMat;
	assert(tensor.dimension(2) <= 1);
	cv::Mat_<_T> mat(tensor.dimension(0), tensor.dimension(1));
	for (_TIndex i = 0; i < tensor.dimension(0); ++i)
		for (_TIndex j = 0; j < tensor.dimension(1); ++j)
			mat(i, j) = tensor(i, j, 0);
	return mat;
}

template <typename _T, int cn, typename _TTensor, int Options>
void copy_mat_tensor(const cv::Mat_<cv::Vec<_T, cn> > &mat, Eigen::TensorMap<_TTensor, Options> tensor)
{
	typedef Eigen::DenseIndex _TIndex;
	for (size_t i = 0; i < tensor.size(); ++i)
		tensor(i) = mat.data[i];
}

template <typename _T, typename _TTensor, int Options>
void copy_mat_tensor(const cv::Mat_<_T> &mat, Eigen::TensorMap<_TTensor, Options> tensor)
{
	typedef Eigen::DenseIndex _TIndex;
	assert(mat.rows == tensor.dimension(0));
	assert(mat.cols == tensor.dimension(1));
	assert(tensor.dimension(2) <= 1);
	for (size_t i = 0; i < tensor.size(); ++i)
		tensor(i) = mat.data[i];
}

template <typename _TTensor, typename _T, int cn>
_TTensor mat_tensor(const cv::Mat_<cv::Vec<_T, cn> > &mat)
{
	_TTensor tensor(mat.rows, mat.cols, mat.channels());
	copy_mat_tensor<_T, cn>(mat, typename tensorflow::TTypes<_T, 3>::Tensor(tensor.data(), tensor.dimensions()));
	return tensor;
}

template <typename _TTensor, typename _T>
_TTensor mat_tensor(const cv::Mat_<_T> &mat)
{
	_TTensor tensor(mat.rows, mat.cols, mat.channels());
	copy_mat_tensor<_T>(mat, typename tensorflow::TTypes<_T, 3>::Tensor(tensor.data(), tensor.dimensions()));
	return tensor;
}

template <typename _TConstTensor, typename _TTensor, int Options>
void copy_tensor(Eigen::TensorMap<_TConstTensor, Options> src, Eigen::TensorMap<_TTensor, Options> dst)
{
	typedef Eigen::DenseIndex _TIndex;
	for (_TIndex i = 0; i < dst.dimension(0); ++i)
		for (_TIndex j = 0; j < dst.dimension(1); ++j)
			for (_TIndex k = 0; k < dst.dimension(2); ++k)
				dst(i, j, k) = src(i, j, k);
}
}
