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
import argparse
import configparser
import collections
import queue
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PyQt4 import QtCore, QtGui
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
import cv2
import pyopenpose
import utils.preprocess


class CameraThread(QtCore.QThread):
    updated = QtCore.pyqtSignal()
    
    def __init__(self, debugger, cache=1, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.debugger = debugger
        self.running = True
        self.queue = queue.Queue(cache)
    
    def run(self):
        try:
            cap = cv2.VideoCapture(args.camera)
            while self.running:
                ret, image_bgr = cap.read()
                assert ret
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = utils.preprocess.resize(image_rgb, self.debugger.size)
                image = utils.preprocess.per_image_standardization(image_rgb)
                tensor = self.debugger.get_current_tensor()
                feature = self.debugger.sess.run(tensor, {self.debugger.image: np.expand_dims(image, 0)})[0]
                self.queue.put((image_rgb, feature))
                self.updated.emit()
        finally:
            cap.release()
    
    def quit(self):
        self.running = False
        QtCore.QThread.wait(self)
        self.queue.join()


class Debugger(QtGui.QWidget):
    def __init__(self, sess, image, size):
        super(Debugger, self).__init__()
        self.sess = sess
        self.image = image
        self.size = size
        layout = QtGui.QVBoxLayout(self)
        
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        
        self.list_tensors = QtGui.QListWidget()
        self.list_tensors.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        layout.addWidget(self.list_tensors)
        
        _layout = QtGui.QHBoxLayout()
        self.lbl_slider = QtGui.QLabel()
        _layout.addWidget(self.lbl_slider)
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        _layout.addWidget(self.slider)
        self.chk_show_image = QtGui.QCheckBox()
        _layout.addWidget(self.chk_show_image)
        layout.addLayout(_layout)
        
        self.tensors = self.load_candidate_tensors(sess)
        for name, _ in self.tensors.items():
            self.list_tensors.addItem(name)
        self.list_tensors.currentItemChanged.connect(self.on_change_tensor)
        self.slider.valueChanged[int].connect(self.on_change_channel)
        
        self.thread = CameraThread(self)
        self.thread.updated.connect(self.update)
        self.thread.start()
    
    def load_candidate_tensors(self, sess):
        tensors = [(op.name, op.values()[0]) for op in tf.get_default_graph().get_operations() if op.values()]
        tensors = [(name, tensor, tensor.get_shape()) for name, tensor in tensors]
        tensors = [(name, tensor, shape.as_list()) for name, tensor, shape in tensors if shape.dims and len(shape.dims) == 4 and shape[0].value == 1 and shape[1].value > 1 and shape[2].value > 1]
        return collections.OrderedDict([(name, dict(tensor=tensor, shape=shape)) for name, tensor, shape in tensors])
    
    def on_change_tensor(self):
        name = self.list_tensors.currentItem().text()
        tensor = self.tensors[name]['tensor']
        self.slider.setSliderPosition(0)
        self.slider.setRange(0, tensor.get_shape().as_list()[3] - 1)
        self.slider.setSliderPosition(0)
        self.slider.valueChanged[int].emit(0)
        self.setWindowTitle(name + ': ' + str(self.tensors[name]['shape']))
    
    def on_change_channel(self, index):
        self.lbl_slider.setText('%d/%d' % (index, self.slider.maximum()))
    
    def get_current_tensor(self):
        name = self.list_tensors.currentItem().text()
        return self.tensors[name]['tensor']
    
    def update(self):
        image_rgb, feature = self.thread.queue.get()
        self.ax.cla()
        assert len(feature.shape) == 3
        feature = feature[:, :, self.slider.value()]
        if self.chk_show_image.checkState() > 0:
            self.ax.imshow(image_rgb)
            feature = cv2.resize(feature, self.size)
        self.ax.imshow(feature, alpha=args.alpha)
        self.canvas.draw()
        matplotlib.pyplot.draw()


def main():
    cachedir = utils.get_cachedir(config)
    logdir = utils.get_logdir(config)
    with open(cachedir + '.parts', 'r') as f:
        num_parts = int(f.read())
    limbs_index = utils.get_limbs_index(config)
    assert pyopenpose.limbs_points(limbs_index) == num_parts
    size_image = config.getint('config', 'height'), config.getint('config', 'width')
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, size_image[0], size_image[1], 3], name='image')
        net = utils.parse_attr(config.get('backbone', 'dnn'))(config, image, train=True)
        utils.parse_attr(config.get('stages', 'dnn'))(config, net, len(limbs_index), num_parts)
        model_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + model_path)
        slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)
        app = QtGui.QApplication(sys.argv)
        widget = Debugger(sess, image, size_image[::-1])
        widget.show()
        sys.exit(app.exec_())


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='model file')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('-i', '--image')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
