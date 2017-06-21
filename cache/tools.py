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
import tensorflow as tf


def image_size(path):
    with Image.open(path) as image:
        return image.size


def verify_image_jpeg(imagepath):
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
    return len(image.shape) == 3


def verify_image_png(imagepath):
    scope = inspect.stack()[0][3]
    try:
        graph = tf.get_default_graph()
        path = graph.get_tensor_by_name(scope + '/path:0')
        decode = graph.get_tensor_by_name(scope + '/decode_png:0')
    except KeyError:
        tf.logging.debug('creating decode_png tensor')
        path = tf.placeholder(tf.string, name=scope + '/path')
        imagefile = tf.read_file(path, name=scope + '/read_file')
        decode = tf.image.decode_png(imagefile, channels=3, name=scope + '/decode_png')
    try:
        image = tf.get_default_session().run(decode, {path: imagepath})
    except:
        return False
    return len(image.shape) == 3
