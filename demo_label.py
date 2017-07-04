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
import configparser
import numpy as np
import scipy.misc
from PyQt4 import QtCore, QtGui
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt4agg
import tensorflow as tf
import utils.data
import utils.visualize


class Visualizer(QtGui.QDialog):
    def __init__(self, name, image, mask, feature):
        super(Visualizer, self).__init__()
        utils.visualize.draw_mask(image, mask.astype(np.uint8) * 255)
        self.name = name
        self.image = image
        self.feature = feature
        
        layout = QtGui.QVBoxLayout(self)
        fig = plt.Figure()
        self.ax = fig.gca()
        self.canvas = matplotlib.backends.backend_qt4agg.FigureCanvasQTAgg(fig)
        layout.addWidget(self.canvas)
        toolbar = matplotlib.backends.backend_qt4agg.NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, feature.shape[2] - 1)
        layout.addWidget(self.slider)
        self.slider.valueChanged[int].connect(self.on_progress)
        
        self.ax.imshow(self.image)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.on_progress(0)

    def on_progress(self, index):
        try:
            self.last.remove()
        except AttributeError:
            pass
        feature = self.feature[:, :, index]
        feature = scipy.misc.imresize(feature, self.image.shape[:2])
        self.last = self.ax.imshow(feature, alpha=args.alpha)
        self.canvas.draw()
        plt.draw()
        self.setWindowTitle('%s %d/%d' % (self.name, index + 1, self.feature.shape[2]))


def main():
    cachedir = utils.get_cachedir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    size_image = config.getint('config', 'height'), config.getint('config', 'width')
    size_label = utils.calc_downsampling_size(config.get('backbone', 'dnn'), size_image[0], size_image[1])
    tf.logging.info('size_image=%s, size_label=%s' % (str(size_image), str(size_label)))
    paths = [os.path.join(cachedir, phase + '.tfrecord') for phase in args.phase]
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
    tf.logging.warn('num_examples=%d' % num_examples)
    with tf.Session() as sess:
        data = utils.data.load_data(config, paths, size_image, size_label, num_parts, limbs_index)
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        while True:
            image, mask, _, limbs, parts = sess.run(data)
            assert image.shape[:2] == size_image
            assert limbs.shape[:2] == size_label
            assert parts.shape[:2] == size_label
            image = image.astype(np.uint8)
            assert limbs.shape[2] == len(limbs_index) * 2
            assert parts.shape[2] == num_parts + 1
            dialog = Visualizer('limbs', image, mask, limbs)
            dialog.exec()
            dialog = Visualizer('parts', image, mask, parts)
            dialog.exec()
        coord.request_stop()
        coord.join(threads)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    app = QtGui.QApplication(sys.argv)
    main()
    sys.exit(app.exec_())
