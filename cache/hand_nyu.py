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

import os
import tqdm
import numpy as np
import scipy.io
import tensorflow as tf
from . import tools


def cache(path, writer, mapper, args, config):
    name = __name__.split('.')[-1]
    cachedir = os.path.dirname(path)
    profile = os.path.splitext(os.path.basename(path))[0]
    profiledir = os.path.join(cachedir, profile)
    os.makedirs(profiledir, exist_ok=True)
    mask_ext = config.get('cache', 'mask_ext')
    with open('config/cache/%s.txt' % name, 'r') as f:
        root = f.read().rstrip()
    root = os.path.join(os.path.expanduser(os.path.expandvars(root)), profile)
    path = os.path.join(root, 'joint_data.mat')
    if not os.path.exists(path):
        tf.logging.warn(path + ' not exists')
        return
    matlab_data = scipy.io.loadmat(path)
    joint_uvd = matlab_data['joint_uvd']
    index_filenames = [(image_index, kinect_index, '%d_%07d' % (image_index + 1, kinect_index + 1)) for kinect_index in range(joint_uvd.shape[1]) for image_index in range(joint_uvd.shape[0])]
    index_filenames = list(filter(lambda item: os.path.exists(os.path.join(root, 'rgb_' + item[-1] + '.png')), index_filenames))
    if len(joint_uvd) > len(index_filenames):
        tf.logging.warn('%d of %d images not exists\n' % (len(joint_uvd) - len(index_filenames), len(joint_uvd)))
    cnt_noobj = 0
    for image_index, kinect_index, filename in tqdm.tqdm(index_filenames):
        # image
        imagepath = os.path.join(root, 'rgb_' + filename + '.png')
        if args.verify:
            if not tools.verify_image_png(imagepath):
                tf.logging.error('failed to decode ' + imagepath)
                continue
        # keypoints
        keypoints = np.squeeze(joint_uvd[image_index, kinect_index, :, :])
        keypoints[:, 2] = 1
        keypoints = np.array(keypoints, dtype=np.int32)
        # mask
        filename = os.path.splitext(os.path.basename(imagepath))[0]
        maskpath = os.path.join(profiledir, filename + '.mask' + mask_ext)
        width, height = tools.image_size(imagepath)
        mask = np.ones(shape=(height, width), dtype=np.uint8) * 255
        scipy.misc.imsave(os.path.join(cachedir, maskpath), mask)
        if args.dump:
            np.save(os.path.join(profiledir, filename + '.npy'), keypoints)
        example = tf.train.Example(features=tf.train.Features(feature={
            'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imagepath)])),
            'maskpath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(maskpath)])),
            'keypoints': tf.train.Feature(bytes_list=tf.train.BytesList(value=[keypoints.tostring()])),
        }))
        writer.write(example.SerializeToString())
    if cnt_noobj > 0:
        tf.logging.warn('%d of %d images have no object' % (cnt_noobj, len(index_filenames)))
