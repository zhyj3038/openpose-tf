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
#include <iostream>
#include <boost/format.hpp>
#include <tensorflow/core/framework/tensor_types.h>
#ifdef DEBUG_SHOW
#include <openpose/debug_global.hpp>
#include <openpose/render.hpp>
#endif

namespace openpose
{
namespace postprocess
{
size_t points_count(const std::vector<Eigen::DenseIndex> &points);
size_t limbs_points(const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > &limbs_index);

template <typename _T>
bool check_connections(const std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &connections)
{
	if (connections.empty())
		return true;
	for (auto i = ++connections.begin(); i != connections.end(); ++i)
	{
		const auto &connection = *i;
		const Eigen::DenseIndex p1 = std::get<0>(connection);
		const Eigen::DenseIndex p2 = std::get<1>(connection);
		for (auto j = connections.begin(); j != i; ++j)
		{
			const auto &_connection = *j;
			if (std::get<0>(_connection) == p1 || std::get<1>(_connection) == p2)
				return false;
		}
	}
	return true;
}

template <typename _T, typename _TTensor, int Options>
std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > calc_limb_score(const Eigen::DenseIndex channel, Eigen::TensorMap<_TTensor, Options> limbs, const std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &peaks1, const std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &peaks2, const size_t steps, const _T min_score, const size_t min_count, const _T epsilon = 1e-6, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef std::tuple<_TIndex, _TIndex, _T> _TConnection;
	typedef std::list<_TConnection> _TConnections;

	assert(min_count > 0);
	assert(epsilon > 0);
	_TConnections connections;
	for (size_t i = 0; i < peaks1.size(); ++i)
	{
		const auto &peak1 = peaks1[i];
		for (size_t j = 0; j < peaks2.size(); ++j)
		{
			const auto &peak2 = peaks2[j];
			const Eigen::DenseIndex y1 = std::get<0>(peak1), x1 = std::get<1>(peak1);
			const Eigen::DenseIndex y2 = std::get<0>(peak2), x2 = std::get<1>(peak2);
			const Eigen::DenseIndex dx = x2 - x1, dy = y2 - y1; //diff
			const _T dist = sqrt((_T)(dx * dx + dy * dy));
			if (dist < epsilon)
				continue;
			const _T nx = dx / dist, ny = dy / dist; //norm
			_T score = 0;
			size_t count = 0;
			for (size_t s = 0; s < steps; ++s)
			{
				const _T prog = (_T)s / steps;
				const _TIndex y = round(y1 + dy * prog);
				const _TIndex x = round(x1 + dx * prog);
				const _T _score = (nx * limbs(y, x, channel) + ny * limbs(y, x, channel + 1));
				if (_score > min_score)
				{
					score += _score;
					++count;
				}
			}
			if (count >= min_count)
				connections.push_back(std::make_tuple(i, j, score / count));
		}
	}
	return connections;
}

template <typename _T>
void filter_connections(std::list<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > &connections, const Eigen::DenseIndex peaks1, const Eigen::DenseIndex peaks2)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef std::tuple<_TIndex, _TIndex, _T> _TConnection;
	typedef std::vector<_TConnection> _TConnections;
	_TConnections _connections(connections.begin(), connections.end());
	std::sort(_connections.begin(), _connections.end(), [](const _TConnection &c1, const _TConnection &c2)->bool{return std::get<2>(c1) > std::get<2>(c2);});
	std::vector<bool> occur1(peaks1, false), occur2(peaks2, false);
	connections.clear();
	const _TIndex num = std::min<_TIndex>(std::min<_TIndex>(peaks1, peaks2), _connections.size());
	for (_TIndex i = 0; i < num; ++i)
	{
		const _TConnection &connection = _connections[i];
		const _TIndex p1 = std::get<0>(connection);
		const _TIndex p2 = std::get<1>(connection);
		if (!occur1[p1] && !occur2[p2])
		{
			connections.push_back(connection);
			occur1[p1] = true;
			occur2[p2] = true;
		}
	}
	assert(check_connections(connections));
}

template <typename _T>
typename std::list<std::tuple<std::vector<Eigen::DenseIndex>, _T, Eigen::DenseIndex> >::iterator find_cluster(const std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> &peak, std::list<std::tuple<std::vector<Eigen::DenseIndex>, _T, Eigen::DenseIndex> > &clusters, const size_t channel, const Eigen::DenseIndex p)
{
	typedef Eigen::DenseIndex _TIndex;
	for (auto c = clusters.begin(); c != clusters.end(); ++c)
	{
		const auto &points = std::get<0>(*c);
		if (points[channel] == p)
			return c;
	}
	return clusters.end();
}

template <typename _T, typename _TTensor, int Options>
std::list<std::tuple<std::vector<Eigen::DenseIndex>, _T, Eigen::DenseIndex> > clustering(const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > &limbs_index, Eigen::TensorMap<_TTensor, Options> limbs, const std::vector<std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > > &peaks, const size_t steps, const _T min_score, const size_t min_count, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_T, 2, Eigen::RowMajor, _TIndex> _TLimb;
	typedef std::vector<Eigen::DenseIndex> _TPoints;
	typedef std::tuple<_TPoints, _T, _TIndex> _TCluster;
	std::list<_TCluster> clusters;
	assert(limbs.dimension(2) == limbs_index.size() * 2);
	assert(limbs_points(limbs_index) == peaks.size());
	for (_TIndex index = 0; index < limbs_index.size(); ++index)
	{
		const std::pair<_TIndex, _TIndex> &limb_index = limbs_index[index];
		assert(0 <= limb_index.first && limb_index.first < peaks.size());
		assert(0 <= limb_index.second && limb_index.second < peaks.size());
		const auto &peaks1 = peaks[limb_index.first];
		const auto &peaks2 = peaks[limb_index.second];
#if 0
		if (peaks1.empty() && peaks2.empty())
			continue;
		if (peaks1.empty())
		{
			for (size_t p = 0; p < peaks2.size(); ++p)
			{
				const auto &peak = peaks2[p];
				if (find_cluster(peak, clusters, limb_index.second, p) == clusters.end())
				{
					_TPoints points(peaks.size(), -1);
					points[limb_index.second] = p;
					clusters.push_back(std::make_tuple(points, std::get<2>(peak), 1));
				}
			}
			continue;
		}
		else if (peaks2.empty())
		{
			for (size_t p = 0; p < peaks1.size(); ++p)
			{
				const auto &peak = peaks1[p];
				if (find_cluster(peak, clusters, limb_index.first, p) == clusters.end())
				{
					_TPoints points(peaks.size(), -1);
					points[limb_index.first] = p;
					clusters.push_back(std::make_tuple(points, std::get<2>(peak), 1));
				}
			}
			continue;
		}
#endif
		const _TIndex channel = index * 2;
		auto connections = calc_limb_score(channel, limbs, peaks1, peaks2, steps, min_score, min_count);
		filter_connections(connections, peaks1.size(), peaks2.size());
#if 1
#ifdef DEBUG_SHOW
		{
			typename tensorflow::TTypes<_T, 3>::ConstTensor _parts_(parts_.data(), parts_.dimensions());
			const auto _canvas = render(image_, limbs_index, _parts_, peaks, clusters);
			const auto canvas = render(_canvas, limb_index, _parts_, peaks, connections);
			const std::string title = (boost::format("before: limb%d (%d-%d)") % index % limb_index.first % limb_index.second).str();
			cv::imshow(title, canvas);
			cv::moveWindow(title, 0, 0);
			std::cout << title << std::endl;
		}
#endif
#endif
		for (auto i = connections.begin(); i != connections.end(); ++i)
		{
			const auto &connection = *i;
			const _TIndex p1 = std::get<0>(connection), p2 = std::get<1>(connection);
			const auto &peak1 = peaks1[p1];
			const auto &peak2 = peaks2[p2];
			{
				auto c = find_cluster(peak1, clusters, limb_index.first, p1);
				if (c != clusters.end())
				{
#if 1
#ifdef DEBUG_SHOW
					std::cout << boost::format("cluster%d (%d->%d)") % std::distance(clusters.begin(), c) % p1 % p2 << std::endl;
#endif
#endif
					auto &cluster = *c;
					auto &points = std::get<0>(cluster);
					assert(points[limb_index.first] == p1);
					//assert(points[limb_index.second] == -1);
					points[limb_index.second] = p2;
					std::get<1>(cluster) += std::get<2>(connection) + std::get<2>(peak2);
					std::get<2>(cluster) += 1;
					continue;
				}
			}
#if 1
#ifdef DEBUG_SHOW
			std::cout << boost::format("cluster%d (%d-%d)") % clusters.size() % p1 % p2 << std::endl;
#endif
#endif
			_TPoints points(peaks.size(), -1);
			points[limb_index.first] = p1;
			points[limb_index.second] = p2;
			clusters.push_back(std::make_tuple(points, std::get<2>(connection) + std::get<2>(peak1) + std::get<2>(peak2), 2));
		}
#if 1
#ifdef DEBUG_SHOW
		{
			const auto canvas = render(image_, limbs_index, typename tensorflow::TTypes<_T, 3>::ConstTensor(parts_.data(), parts_.dimensions()), peaks, clusters);
			const std::string title = (boost::format("after: limb%d (%d-%d)") % index % limb_index.first % limb_index.second).str();
			cv::imshow(title, canvas);
			cv::moveWindow(title, canvas.cols, 0);
			std::cout << title << std::endl;
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
#endif
#endif
	}
	return clusters;
}

template <typename _T>
std::list<std::list<std::pair<std::pair<_T, _T>, std::pair<_T, _T> > > > filter_cluster(const std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex> > &limbs_index, const std::vector<std::vector<std::tuple<Eigen::DenseIndex, Eigen::DenseIndex, _T> > > &peaks, const std::list<std::tuple<std::vector<Eigen::DenseIndex>, _T, Eigen::DenseIndex> > &clusters, const _T min_score, const size_t min_count)
{
	typedef Eigen::DenseIndex _TIndex;
	typedef std::pair<_T, _T> _TPoint;
	typedef std::pair<_TPoint, _TPoint> _TEdge;
	typedef std::list<_TEdge> _TKeypoints;
	std::list<_TKeypoints> results;
	assert(min_count > 0);
	for (auto c = clusters.begin(); c != clusters.end(); ++c)
	{
		const auto &cluster = *c;
		const _TIndex count = std::get<2>(cluster);
		if (count >= min_count && std::get<1>(cluster) / count > min_score)
		{
			results.push_back(_TKeypoints());
			auto &keypoints = results.back();
			const std::vector<_TIndex> &points = std::get<0>(cluster);
			for (size_t l = 0; l < limbs_index.size(); ++l)
			{
				const std::pair<_TIndex, _TIndex> &limb_index = limbs_index[l];
				const _TIndex p1 = points[limb_index.first], p2 = points[limb_index.second];
				if (p1 >= 0 && p2 >= 0)
				{
					const auto &_p1 = peaks[limb_index.first][p1];
					const auto &_p2 = peaks[limb_index.second][p2];
					keypoints.push_back(std::make_pair(std::make_pair(std::get<0>(_p1), std::get<1>(_p1)), std::make_pair(std::get<0>(_p2), std::get<1>(_p2))));
				}
			}
		}
	}
	return results;
}
}
}
