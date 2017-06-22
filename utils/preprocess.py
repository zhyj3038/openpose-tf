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

import numpy as np
from PIL import Image, ExifTags
import cv2


def per_image_standardization(image):
    stddev = np.std(image)
    return (image - np.mean(image)) / max(stddev, 1.0 / np.sqrt(np.multiply.reduce(image.shape)))


def read_image(path):
    image = Image.open(path)
    for key in ExifTags.TAGS.keys():
        if ExifTags.TAGS[key] == 'Orientation':
            break
    try:
        exif = dict(image._getexif().items())
    except AttributeError:
        return image
    if exif[key] == 3:
        image = image.rotate(180, expand=True)
    elif exif[key] == 6:
        image = image.rotate(270, expand=True)
    elif exif[key] == 8:
        image = image.rotate(90, expand=True)
    return image


def resize(image, size):
    height, width, _ = image.shape
    _width, _height = size
    if height / width > _height / _width:
        scale = _height / height
    else:
        scale = _width / width
    m = np.eye(2, 3)
    m[0, 0] = scale
    m[1, 1] = scale
    return cv2.warpAffine(image, m, size, flags=cv2.INTER_CUBIC)
