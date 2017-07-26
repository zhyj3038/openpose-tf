# -*- coding: utf-8 -*-

"""
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
"""

import sys
import os
import argparse
import h5py
import caffe
from caffe.proto import caffe_pb2


def main():
    net = caffe_pb2.NetParameter()
    path = os.path.expanduser(os.path.expandvars(args.path))
    sys.stderr.write('load ' + path + '\n')
    with open(path, 'r') as f:
        net.ParseFromString(f.read())
    path = os.path.splitext(path)[0] + '.h5'
    sys.stderr.write('save ' + path + '\n')
    with h5py.File(path, 'w') as f:
        for layer in net.layer:
            name = layer.name
            grp = f.create_group(name)
            for i, blob in enumerate(layer.blobs):
                var = caffe.io.blobproto_to_array(blob)
                print('%s/%d\t%s' % (name, i, ','.join(map(str, var.shape))))
                grp.create_dataset(str(i), data=var)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='the .caffemodel file')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    main()
