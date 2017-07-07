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

#include <utility>
#include <vector>
#include <list>
#include "nms.hpp"
#include "hungarian.hpp"

namespace openpose
{
namespace postprocess
{
template <typename _T, typename _TTensor, int Options>
std::list<std::list<std::pair<std::tuple<Eigen::DenseIndex, _T, _T>, std::tuple<Eigen::DenseIndex, _T, _T> > > > estimate(const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > &limbs_index, Eigen::TensorMap<_TTensor, Options> limbs, Eigen::TensorMap<_TTensor, Options> parts, const _T threshold, const size_t limits, const size_t steps, const _T min_score, const size_t min_count, const _T cluster_min_score, const size_t cluster_min_count)
{
	const auto peaks = featuremap_peaks(parts, threshold, limits);
	auto clusters = clustering(limbs_index, limbs, peaks, steps, min_score, min_count);
	return filter_cluster(limbs_index, peaks, clusters, cluster_min_score, cluster_min_count);
}
}
}
