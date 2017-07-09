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

#include <vector>
#include <list>
#include <tuple>
#include <cmath>
#include <type_traits>
#include <tensorflow/core/framework/tensor_types.h>

namespace openpose
{
namespace postprocess
{
template <typename _TTensor, int Options>
std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, typename std::remove_const<typename _TTensor::Scalar>::type> > feature_peaks(Eigen::TensorMap<_TTensor, Options> feature, const typename _TTensor::Scalar threshold)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename std::remove_const<typename _TTensor::Scalar>::type _T;
	typedef std::tuple<_TIndex, _TIndex, _T> _TPeak;
	typedef std::list<_TPeak> _TPeaks;
	_TPeaks peaks;
	for (_TIndex y = 0; y < feature.dimension(0); ++y)
		for (_TIndex x = 0; x < feature.dimension(1); ++x)
		{
			const _T value = feature(y, x);
			if (value < threshold)
				continue;
			const _T top = y == 0 ? 0 : feature(y - 1, x);
			const _T bottom = y == feature.dimension(0) - 1 ? 0 : feature(y + 1, x);
			const _T left = x == 0 ? 0 : feature(y, x - 1);
			const _T right = x == feature.dimension(1) - 1 ? 0 : feature(y, x + 1);
			if(value > top && value > bottom && value > left && value > right)
				peaks.push_back(std::make_tuple(y, x, value));
		}
	return peaks;
}

template <typename _T>
std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > filter_peaks(std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &peaks, const _T radius)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef std::tuple<_TIndex, _TIndex, _T> _TPeak;
	typedef std::list<_TPeak> _TPeaks;
	typedef typename _TPeaks::iterator _TIterator;
	std::vector<_TIterator> _peaks(peaks.size());
	{
		size_t index = 0;
		for (_TIterator i = peaks.begin(); i != peaks.end(); ++i)
		{
			_peaks[index] = i;
			++index;
		}
	}
	std::sort(_peaks.begin(), _peaks.end(), [](_TIterator a, _TIterator b)->bool{return std::get<2>(*a) > std::get<2>(*b);});
	for (size_t i = 0; i < _peaks.size(); ++i)
	{
		const _TPeak &peak1 = *_peaks[i];
		if (std::get<2>(peak1) <= 0)
			continue;
		const _T y1 = std::get<0>(peak1), x1 = std::get<1>(peak1);
		for (size_t j = i + 1; j < _peaks.size(); ++j)
		{
			_TPeak &peak2 = *_peaks[j];
			if (std::get<2>(peak2) <= 0)
				continue;
			const _T y2 = std::get<0>(peak2), x2 = std::get<1>(peak2);
			const _T xdiff = x1 - x2, ydiff = y1 - y2;
			const _T dist = sqrt(xdiff * xdiff + ydiff * ydiff);
			if (dist < radius)
				std::get<2>(peak2) = 0;
		}
	}
	for (_TIterator i = peaks.begin(); i != peaks.end();)
	{
		const _TPeak &peak = *i;
		if (std::get<2>(peak) <= 0)
		{
			_TIterator remove = i;
			++i;
			peaks.erase(remove);
		}
		else
			++i;
	}
	return std::vector<_TPeak>(peaks.begin(), peaks.end());
}

template <typename _TTensor, int Options>
std::vector<std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, typename std::remove_const<typename _TTensor::Scalar>::type> > > featuremap_peaks(Eigen::TensorMap<_TTensor, Options> featuremap, const typename _TTensor::Scalar threshold, const typename _TTensor::Scalar radius)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename std::remove_const<typename _TTensor::Scalar>::type _T;
	typedef std::tuple<_TIndex, _TIndex, _T> _TPeak;
	typedef std::vector<_TPeak> _TPeaks;
	std::vector<_TPeaks> result(featuremap.dimension(2));
	for (size_t i = 0; i < result.size(); ++i)
	{
		const Eigen::Tensor<_T, 2, Eigen::RowMajor, _TIndex> feature = featuremap.chip(i, 2);
		auto peaks = feature_peaks(typename tensorflow::TTypes<_T, 2>::ConstTensor(feature.data(), feature.dimensions()), threshold);
		result[i] = filter_peaks(peaks, radius);
	}
	return result;
}
}
}
