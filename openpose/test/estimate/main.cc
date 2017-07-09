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

#ifndef NDEBUG
#define DEBUG_SHOW
#endif

#include <boost/format.hpp>
#include <Eigen/Core>
#include <tensorflow/core/framework/tensor_types.h>
#include <tensorflow/core/platform/default/integral_types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <openpose/postprocess/nms.hpp>
#include <openpose/postprocess/hungarian.hpp>
#include <openpose/npy.hpp>
#include <openpose/tsv.hpp>
#include <openpose/render.hpp>

int main(void)
{
	typedef uchar _TPixel;
	typedef float _TReal;
	typedef Eigen::DenseIndex _TIndex;
	typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex> _TTensorReal;
	typedef std::pair<Eigen::DenseIndex, Eigen::DenseIndex> _TLimbIndex;
	typedef std::vector<_TLimbIndex> _TLimbsIndex;
	typedef cv::Vec<_TPixel, 3> _TVec3;
	typedef cv::Mat_<_TVec3> _TMat3;

	const _TReal threshold = 0.05;
	const _TReal radius = 7;
	const size_t steps = 10;
	const _TReal min_score = 0.05;
	const size_t min_count = 9;
	const _TMat3 image = cv::imread(DUMP_DIR "/featuremap/image.jpg", CV_LOAD_IMAGE_COLOR);
	const auto limbs_index = openpose::load_tsv_paired<Eigen::DenseIndex>(DUMP_DIR "/featuremap/limbs_index.tsv");
	const _TTensorReal limbs = openpose::load_npy3<_TReal, _TTensorReal>(DUMP_DIR "/featuremap/limbs.npy");
	const _TTensorReal parts = openpose::load_npy3<_TReal, _TTensorReal>(DUMP_DIR "/featuremap/parts.npy");
#ifdef DEBUG_SHOW
	openpose::image_ = image;
	openpose::parts_ = parts;
#endif
	const auto peaks = openpose::postprocess::featuremap_peaks(typename tensorflow::TTypes<_TReal, 3>::ConstTensor(parts.data(), parts.dimensions()), threshold, radius);
	const auto clusters = openpose::postprocess::clustering(_TLimbsIndex(limbs_index.begin(), limbs_index.end()), typename tensorflow::TTypes<_TReal, 3>::ConstTensor(limbs.data(), limbs.dimensions()), peaks, steps, min_score, min_count);
	return 0;
}
