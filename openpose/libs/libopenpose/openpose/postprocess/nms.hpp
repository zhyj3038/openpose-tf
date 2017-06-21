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
std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > limit_peaks(const std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &peaks, const size_t limits)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef std::tuple<_TIndex, _TIndex, _T> _TPeak;
	typedef std::vector<_TPeak> _TPeaks;
	_TPeaks _peaks(peaks.begin(), peaks.end());
	const size_t _limits = std::min(_peaks.size(), limits);
	std::partial_sort(_peaks.begin(), _peaks.begin() + _limits, _peaks.end(), [](const _TPeak &a, const _TPeak &b)->bool{return std::get<2>(a) < std::get<2>(b);});
	_peaks.resize(_limits);
	return _peaks;
}
}
}
