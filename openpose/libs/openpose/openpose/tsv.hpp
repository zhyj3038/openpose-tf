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

#include <fstream>
#include <vector>
#include <list>
#include <type_traits>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/exception/all.hpp>

namespace openpose
{
template <typename _TTensor>
_TTensor load_tsv_tensor(const std::string &path, const char delimiter = '\t', typename std::enable_if<_TTensor::NumIndices == 2>::type* = nullptr)
{
	typedef typename _TTensor::Scalar _TScalar;
	typedef typename _TTensor::Index _TIndex;
	typedef std::vector<_TScalar> _TRow;
	typedef boost::char_delimiters_separator<char> _TSeparator;
	typedef boost::tokenizer<_TSeparator> _TTokenizer;

	std::list<_TRow> data;
	std::ifstream fs(path.c_str());
	std::string line;
	_TSeparator sep(delimiter);
	while (getline(fs, line))
	{
		_TTokenizer tok(line, sep);
		const std::vector<std::string> _row(tok.begin(), tok.end());
		_TRow row(_row.size());
		for (size_t i = 0; i < _row.size(); ++i)
			row[i] = boost::lexical_cast<_TScalar>(_row[i]);
		data.push_back(row);
		assert(row.size() == data.front().size());
	}
	_TTensor tensor(data.size(), data.front().size());
	size_t index = 0;
	for (auto i = data.begin(); i != data.end(); ++i)
	{
		const _TRow &row = *i;
		for (size_t j = 0; j < row.size(); ++j)
			tensor(index, j) = row[j];
		++index;
	}
	return tensor;
}

template <typename _T>
std::list<std::pair<_T, _T> > load_tsv_paired(const std::string &path, const char delimiter = '\t')
{
	typedef std::pair<_T, _T> _TRow;
	typedef boost::char_delimiters_separator<char> _TSeparator;
	typedef boost::tokenizer<_TSeparator> _TTokenizer;

	std::list<_TRow> data;
	std::ifstream fs(path.c_str());
	std::string line;
	_TSeparator sep(delimiter);
	while (getline(fs, line))
	{
		_TTokenizer tok(line, sep);
		const std::vector<std::string> _row(tok.begin(), tok.end());
		if (_row.size() != 2)
			BOOST_THROW_EXCEPTION(std::runtime_error(line));
		data.push_back(std::make_pair(boost::lexical_cast<_T>(_row[0]), boost::lexical_cast<_T>(_row[1])));
	}
	return data;
}
}

