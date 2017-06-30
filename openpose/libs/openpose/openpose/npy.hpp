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

#include <vector>
#include <cnpy.h>

namespace openpose
{
template <typename _TData, typename _TTensor, int Options>
void save_npy(Eigen::TensorMap<_TTensor, Options> tensor, const std::string &path)
{
	std::vector<unsigned int> shape(tensor.rank());
	for (size_t i = 0; i < shape.size(); ++i)
		shape[i] = tensor.dimension(i);
	std::vector<_TData> data(tensor.size());
	for (size_t i = 0; i < data.size(); ++i)
		data[i] = tensor(i);
	cnpy::npy_save(path, &data.front(), &shape.front(), shape.size(), "w");
}

template <typename _TData, typename _TTensor>
_TTensor load_npy3(const std::string &path)
{
	cnpy::NpyArray arr = cnpy::npy_load(path);
	assert(arr.shape.size() == 3);
	_TTensor tensor(arr.shape[0], arr.shape[1], arr.shape[2]);
	const _TData *data = reinterpret_cast<const _TData *>(arr.data);
	for (size_t i = 0; i < tensor.size(); ++i)
		tensor(i) = data[i];
	arr.destruct();
	assert(tensor.rank() == arr.shape.size());
	return tensor;
}
}
