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

#include "nms.hpp"
#include "hungarian.hpp"
#include "estimate.hpp"

#define QUOTE(x) #x

PYBIND11_PLUGIN(PROJECT_NAME) {
	pybind11::module m(QUOTE(PROJECT_NAME));
	//nms
	m.def("feature_peaks", &feature_peaks<float>);
	m.def("feature_peaks", &feature_peaks<double>);
	m.def("featuremap_peaks", &featuremap_peaks<float>);
	m.def("featuremap_peaks", &featuremap_peaks<double>);
	//hungarian
	m.def("limbs_points", &openpose::postprocess::limbs_points);
	m.def("calc_limb_score", &calc_limb_score<float>);
	m.def("calc_limb_score", &calc_limb_score<double>);
	m.def("filter_connections", &openpose::postprocess::filter_connections<float>);
	m.def("filter_connections", &openpose::postprocess::filter_connections<double>);
	m.def("clustering", &clustering<float>);
	m.def("clustering", &clustering<double>);
	m.def("filter_cluster", &openpose::postprocess::filter_cluster<float>);
	m.def("filter_cluster", &openpose::postprocess::filter_cluster<double>);
	//estimate
	m.def("estimate", &estimate<float>);
	m.def("estimate", &estimate<double>);
	return m.ptr();
}
