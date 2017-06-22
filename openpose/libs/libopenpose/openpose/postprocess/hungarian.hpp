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
#include <vector>
#include <list>
#include <tuple>
#include <type_traits>
#include <tensorflow/core/framework/tensor_types.h>

namespace openpose
{
namespace postprocess
{
template <typename _T, typename _TTensor, int Options>
std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > calc_limb_score(const std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > peaks1, const std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > peaks2, Eigen::TensorMap<_TTensor, Options> limb_x, Eigen::TensorMap<_TTensor, Options> limb_y, const size_t steps, const _T min_score, const size_t min_count)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef std::tuple<_TIndex, _TIndex, _T> _TConnection;
	typedef std::list<_TConnection> _TConnections;

	assert(min_count > 0);
	_TConnections connections;
	for (size_t i = 0; i < peaks1.size(); ++i)
	{
		const auto &_part1 = peaks1[i];
		for (size_t j = 0; j < peaks2.size(); ++j)
		{
			const auto &_part2 = peaks2[j];
			const Eigen::DenseIndex y1 = std::get<0>(_part1), x1 = std::get<1>(_part1);
			const Eigen::DenseIndex y2 = std::get<0>(_part2), x2 = std::get<1>(_part2);
			const Eigen::DenseIndex dx = x2 - x1, dy = y2 - y1; //diff
			const _T dist = sqrt((_T)(dx * dx + dy * dy));
			if (dist < 1e-6)
				continue;
			const _T nx = dx / dist, ny = dy / dist; //norm
			_T score = 0;
			size_t count = 0;
			for (size_t s = 0; s < steps; ++s)
			{
				const _T prog = (_T)s / steps;
				const _TIndex y = round(y1 + dy * prog);
				const _TIndex x = round(x1 + dx * prog);
				const _T _score = (nx * limb_x(y, x) + ny * limb_y(y, x));
				if (_score > min_score)
				{
					score += _score;
					++count;
				}
			}
			if (count > min_count)
				connections.push_back(std::make_tuple(i, j, score / count));
		}
	}
	return connections;
}
}
}
