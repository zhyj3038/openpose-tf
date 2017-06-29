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

#include <set>
#include <algorithm>
#include <numeric>
#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include "hungarian.hpp"

namespace openpose
{
namespace postprocess
{
size_t points_count(const std::vector<Eigen::DenseIndex> &points)
{
	size_t count = 0;
	for (size_t i = 0; i < points.size(); ++i)
		if (points[i] >= 0)
			++count;
	return count;
}

bool check_unique(std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > limbs_index)
{
	std::sort(limbs_index.begin(), limbs_index.end());
	return std::unique(limbs_index.begin(), limbs_index.end()) == limbs_index.end();
}

size_t limbs_points(const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > &limbs_index)
{
	typedef Eigen::DenseIndex _TIndex;
	if (limbs_index.empty())
		throw std::runtime_error("limbs_index is empty");
	if (!check_unique(limbs_index))
		throw std::runtime_error("duplicated limbs found");
	std::set<_TIndex> set;
	set.insert(limbs_index.front().first);
	set.insert(limbs_index.front().second);
	for (size_t i = 1; i < limbs_index.size(); ++i)
	{
		const auto &limb_index = limbs_index[i];
		if (set.find(limb_index.first) == set.end())
			throw std::runtime_error((boost::format("first limb part %d not found in part set {%d}") % limb_index.first % boost::algorithm::join(set | boost::adaptors::transformed(static_cast<std::string(*)(_TIndex)>(std::to_string)), ", ")).str());
		set.insert(limb_index.first);
		set.insert(limb_index.second);
	}
	return set.size();
}
}
}
