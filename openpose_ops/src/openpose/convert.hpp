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
#include <tensorflow/core/framework/tensor_types.h>
#include <opencv2/opencv.hpp>

namespace openpose
{
template <typename _T, int cn>
cv::Mat_<cv::Vec<_T, cn> > tensor_mat(typename tensorflow::TTypes<_T, 3>::ConstTensor tensor)
{
	typedef typename tensorflow::TTypes<_T>::Tensor::Index _TIndex;
	typedef cv::Vec<_T, cn> _TPixel;
	typedef cv::Mat_<_TPixel> _TMat;
	assert(tensor.dimension(2) == cn);
	cv::Mat_<cv::Vec<_T, cn> > mat(tensor.dimension(0), tensor.dimension(1));
	for (_TIndex i = 0; i < tensor.dimension(0); ++i)
		for (_TIndex j = 0; j < tensor.dimension(1); ++j)
		{
			_TPixel &pixel = mat(i, j);
			for (_TIndex k = 0; k < tensor.dimension(2); ++k)
				pixel[k] = tensor(i, j, k);
		}
	return mat;
}

template <typename _T>
cv::Mat_<_T> tensor_mat(typename tensorflow::TTypes<_T, 3>::ConstTensor tensor)
{
	typedef typename tensorflow::TTypes<_T>::Tensor::Index _TIndex;
	typedef cv::Mat_<_T> _TMat;
	assert(tensor.dimension(2) == 1);
	cv::Mat_<_T> mat(tensor.dimension(0), tensor.dimension(1));
	for (_TIndex i = 0; i < tensor.dimension(0); ++i)
		for (_TIndex j = 0; j < tensor.dimension(1); ++j)
			mat(i, j) = tensor(i, j, 0);
	return mat;
}

template <typename _T, int cn>
void copy_mat_tensor(const cv::Mat_<cv::Vec<_T, cn> > &mat, typename tensorflow::TTypes<_T, 3>::Tensor tensor)
{
	typedef typename tensorflow::TTypes<_T>::Tensor::Index _TIndex;
	typedef cv::Vec<_T, cn> _TPixel;
	typedef cv::Mat_<_TPixel> _TMat;
	for (_TIndex i = 0; i < tensor.dimension(0); ++i)
		for (_TIndex j = 0; j < tensor.dimension(1); ++j)
		{
			const _TPixel &pixel = mat(i, j);
			for (_TIndex k = 0; k < tensor.dimension(2); ++k)
				tensor(i, j, k) = pixel[k];
		}
}

template <typename _T>
void copy_mat_tensor(const cv::Mat_<_T> &mat, typename tensorflow::TTypes<_T, 3>::Tensor tensor)
{
	typedef typename tensorflow::TTypes<_T>::Tensor::Index _TIndex;
	typedef cv::Mat_<_T> _TMat;
	assert(tensor.dimension(2) == 1);
	for (_TIndex i = 0; i < tensor.dimension(0); ++i)
		for (_TIndex j = 0; j < tensor.dimension(1); ++j)
			tensor(i, j, 0) = mat(i, j);
}

template <typename _T>
void copy_tensor(typename tensorflow::TTypes<_T, 3>::ConstTensor src, typename tensorflow::TTypes<_T, 3>::Tensor dst)
{
	typedef typename tensorflow::TTypes<_T>::Tensor::Index _TIndex;
	for (_TIndex i = 0; i < dst.dimension(0); ++i)
		for (_TIndex j = 0; j < dst.dimension(1); ++j)
			for (_TIndex k = 0; k < dst.dimension(2); ++k)
				dst(i, j, k) = src(i, j, k);
}
}
