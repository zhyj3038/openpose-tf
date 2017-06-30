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
from PIL import Image, ImageDraw
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import pycocotools.coco
import pycocotools.mask
from . import tools


def draw_mask(segmentation, canvas):
    pixels = canvas.load()
    if isinstance(segmentation, list):
        for polygon in segmentation:
            ImageDraw.Draw(canvas).polygon(polygon, fill=0)
    else:
        if isinstance(segmentation['counts'], list):
            rle = pycocotools.mask.frPyObjects([segmentation], canvas.size[1], canvas.size[0])
        else:
            rle = [segmentation]
        m = np.squeeze(pycocotools.mask.decode(rle))
        assert m.shape[:2] == canvas.size[::-1]
        for y, row in enumerate(m):
            for x, v in enumerate(row):
                if v:
                    pixels[x, y] = 0


def cache(path, writer, mapper, args, config):
    name = __name__.split('.')[-1]
    cachedir = os.path.dirname(path)
    phase = os.path.splitext(os.path.basename(path))[0]
    phasedir = os.path.join(cachedir, phase)
    os.makedirs(phasedir, exist_ok=True)
    mask_ext = config.get('cache', 'mask_ext')
    for i, row in pd.read_csv(os.path.splitext(__file__)[0] + '.tsv', sep='\t').iterrows():
        tf.logging.info('loading data %d (%s)' % (i, ', '.join([k + '=' + str(v) for k, v in row.items()])))
        root = os.path.expanduser(os.path.expandvars(row['root']))
        year = str(row['year'])
        suffix = phase + year
        path = os.path.join(root, 'annotations', 'person_keypoints_%s.json' % suffix)
        if not os.path.exists(path):
            tf.logging.warn(path + ' not exists')
            continue
        coco_kp = pycocotools.coco.COCO(path)
        skeleton = np.array(coco_kp.loadCats(1)[0]['skeleton']) - 1
        np.savetxt(os.path.join(os.path.dirname(cachedir), name + '.tsv'), skeleton, fmt='%d', delimiter='\t')
        imgIds = coco_kp.getImgIds()
        path = os.path.join(root, suffix)
        imgs = coco_kp.loadImgs(imgIds)
        _imgs = list(filter(lambda img: os.path.exists(os.path.join(path, img['file_name'])), imgs))
        if len(imgs) > len(_imgs):
            tf.logging.warn('%d of %d images not exists' % (len(imgs) - len(_imgs), len(imgs)))
        cnt_noobj = 0
        for img in tqdm.tqdm(_imgs):
            # image
            imagepath = os.path.join(path, img['file_name'])
            width, height = img['width'], img['height']
            if args.verify:
                if not np.all(np.equal(tools.image_size(imagepath), [width, height])):
                    tf.logging.error('failed to verify shape of image ' + imagepath)
                    continue
                if not tools.verify_image_jpeg(imagepath):
                    tf.logging.error('failed to decode ' + imagepath)
                    continue
            # keypoints
            annIds = coco_kp.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco_kp.loadAnns(annIds)
            keypoints = []
            filename = os.path.splitext(os.path.basename(imagepath))[0]
            maskpath = os.path.join(phasedir, filename + '.mask' + mask_ext)
            with Image.new('L', (width, height), 255) as canvas:
                for ann in anns:
                    points = mapper(np.array(ann['keypoints']).reshape([-1, 3]))
                    if np.any(points[:, 2] > 0):
                        keypoints.append(points)
                    else:
                        draw_mask(ann['segmentation'], canvas)
                if len(keypoints) <= 0:
                    cnt_noobj += 1
                    continue
                canvas.save(os.path.join(cachedir, maskpath))
            keypoints = np.array(keypoints, dtype=np.int32)
            if args.dump:
                np.save(os.path.join(phasedir, filename + '.npy'), keypoints)
            example = tf.train.Example(features=tf.train.Features(feature={
                'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imagepath)])),
                'maskpath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(maskpath)])),
                'keypoints': tf.train.Feature(bytes_list=tf.train.BytesList(value=[keypoints.tostring()])),
            }))
            writer.write(example.SerializeToString())
        if cnt_noobj > 0:
            tf.logging.warn('%d of %d images have no object' % (cnt_noobj, len(_imgs)))
