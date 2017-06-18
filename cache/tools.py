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

import inspect
from PIL import Image
import numpy as np
import tensorflow as tf


def verify_imageshape(imagepath, imageshape):
    with Image.open(imagepath) as image:
        return np.all(np.equal(image.size, imageshape[1::-1]))


def verify_image_jpeg(imagepath, imageshape):
    scope = inspect.stack()[0][3]
    try:
        graph = tf.get_default_graph()
        path = graph.get_tensor_by_name(scope + '/path:0')
        decode = graph.get_tensor_by_name(scope + '/decode_jpeg:0')
    except KeyError:
        tf.logging.debug('creating decode_jpeg tensor')
        path = tf.placeholder(tf.string, name=scope + '/path')
        imagefile = tf.read_file(path, name=scope + '/read_file')
        decode = tf.image.decode_jpeg(imagefile, channels=3, name=scope + '/decode_jpeg')
    try:
        image = tf.get_default_session().run(decode, {path: imagepath})
    except:
        return False
    return np.all(np.equal(image.shape[:2], imageshape[:2]))


def check_coords(objects_coord):
    return np.all(objects_coord[:, 0] <= objects_coord[:, 2]) and np.all(objects_coord[:, 1] <= objects_coord[:, 3])


def verify_coords(objects_coord, imageshape):
    assert check_coords(objects_coord)
    return np.all(objects_coord >= 0) and np.all(objects_coord <= np.tile(imageshape[1::-1], [2]))


def fix_coords(objects_coord, imageshape):
    assert check_coords(objects_coord)
    objects_coord = np.maximum(objects_coord, np.zeros([4], dtype=objects_coord.dtype))
    objects_coord = np.minimum(objects_coord, np.tile(np.asanyarray(imageshape[1::-1], objects_coord.dtype), [2]))
    return objects_coord
