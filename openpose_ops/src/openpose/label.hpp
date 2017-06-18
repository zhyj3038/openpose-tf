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
#include <cmath>
#include <type_traits>
#include <tensorflow/core/framework/tensor_types.h>

namespace openpose
{
template <typename _T>
std::pair<_T, _T> calc_norm_vec(const _T x, const _T y)
{
	const _T len = sqrt(x * x + y * y);
	return std::make_pair(x / len, y / len);
}

template <typename _TConstTensorReal, typename _TConstTensorInteger, typename _TTensor, int Options>
void make_label(Eigen::TensorMap<_TConstTensorReal, Options> keypoints, Eigen::TensorMap<_TConstTensorInteger, Options> limbs,
	const typename _TConstTensorReal::Scalar sigma_limbs, const typename _TConstTensorReal::Scalar sigma_parts,
	const typename _TConstTensorInteger::Scalar height, const typename _TConstTensorInteger::Scalar width,
	Eigen::TensorMap<_TTensor, Options> label, const typename _TConstTensorReal::Scalar threshold = -log(0.01))
{
	typedef Eigen::DenseIndex _TIndex;
	typedef typename std::remove_const<typename _TConstTensorReal::Scalar>::type _TReal;
	typedef typename std::remove_const<typename _TConstTensorInteger::Scalar>::type _TInteger;
	typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> _TMatrixCount;

	assert(limbs.dimension(1) == 2);
	for (_TIndex gy = 0; gy < label.dimension(0); ++gy)
		for (_TIndex gx = 0; gx < label.dimension(1); ++gx)
			for (_TIndex channel = 0; channel < label.dimension(2) - 1; ++channel)
				label(gy, gx, channel) = 0;
	const _TReal grid_height = (_TReal)height / label.dimension(0);
	const _TReal grid_width = (_TReal)width / label.dimension(1);
	const _TReal grid_height2 = grid_height / 2;
	const _TReal grid_width2 = grid_width / 2;
	for (_TIndex limb = 0; limb < limbs.dimension(0); ++limb)
	{
		const _TInteger channel_x = limb * 2;
		const _TInteger channel_y = channel_x + 1;
		const _TInteger p1 = limbs(limb, 0);
		const _TInteger p2 = limbs(limb, 1);
		_TMatrixCount count = _TMatrixCount::Constant(label.dimension(0), label.dimension(1), 0);
		for (_TIndex index = 0; index < keypoints.dimension(0); ++index)
			if (keypoints(index, p1, 2) > 0 && keypoints(index, p2, 2) > 0)
			{
				const _TReal p1x = keypoints(index, p1, 0), p1y = keypoints(index, p1, 1);
				const _TReal p2x = keypoints(index, p2, 0), p2y = keypoints(index, p2, 1);
				const std::pair<_TReal, _TReal> norm_vec = calc_norm_vec(p2x - p1x, p2y - p1y);
				const std::pair<_TReal, _TReal> range_x = std::minmax(p1x, p2x);
				const std::pair<_TReal, _TReal> range_y = std::minmax(p1y, p2y);
				const _TIndex gx_min = round((range_x.first - sigma_limbs) / grid_width);
				const _TIndex gx_max = round((range_x.second + sigma_limbs) / grid_width);
				const _TIndex gy_min = round((range_y.first - sigma_limbs) / grid_height);
				const _TIndex gy_max = round((range_y.second + sigma_limbs) / grid_height);
				for (_TIndex gy = std::max<_TIndex>(gy_min, 0); gy < std::min(gy_max, label.dimension(0)); ++gy)
					for (_TIndex gx = std::max<_TIndex>(gx_min, 0); gx < std::min(gx_max, label.dimension(1)); ++gx)
					{
						const _TReal x = gx * grid_width + grid_width2;
						const _TReal y = gy * grid_height + grid_height2;
						const _TReal vx = x - p1x, vy = y - p1y;
						const _TReal dist = std::abs(vx * norm_vec.second - vy * norm_vec.first);
						if (dist <= sigma_limbs)
						{
							label(gy, gx, channel_x) += vx;
							label(gy, gx, channel_y) += vy;
							count(gy, gx) += 1;
						}
					}
			}
		for (_TIndex gy = 0; gy < label.dimension(0); ++gy)
			for (_TIndex gx = 0; gx < label.dimension(1); ++gx)
			{
				const size_t c = count(gy, gx);
				if (c > 0)
				{
					label(gy, gx, channel_x) /= c;
					label(gy, gx, channel_y) /= c;
				}
			}
	}
	const _TIndex offset = limbs.dimension(0) * 2;
	const _TReal sigma_parts2 = sigma_parts * sigma_parts;
	for (_TIndex gy = 0; gy < label.dimension(0); ++gy)
		for (_TIndex gx = 0; gx < label.dimension(1); ++gx)
		{
			const _TReal x = gx * grid_width + grid_width2;
			const _TReal y = gy * grid_height + grid_height2;
			_TReal maximum = 0;
			for (_TIndex part = 0; part < keypoints.dimension(1); ++part)
			{
				const _TIndex channel = offset + part;
				for (_TIndex index = 0; index < keypoints.dimension(0); ++index)
					if (keypoints(index, part, 2) > 0)
					{
						const _TReal diff_x = keypoints(index, part, 0) - x;
						const _TReal diff_y = keypoints(index, part, 1) - y;
						const _TReal exponent = (diff_x * diff_x + diff_y * diff_y) / 2 / sigma_parts2;
						if(exponent > threshold)
							continue;
						label(gy, gx, channel) += exp(-exponent);
						if (label(gy, gx, channel) > 1)
							label(gy, gx, channel) = 1;
						maximum = std::max<_TReal>(label(gy, gx, channel), maximum);
					}
				assert(0 <= label(gy, gx, channel) <= 1);
			}
			assert(0 <= maximum <= 1);
			label(gy, gx, label.dimension(2) - 1) = 1 - maximum;
		}
}
}
