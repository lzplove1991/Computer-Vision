#-*- coding:utf-8 -*-

# @File   : SSD300.py
# @Date   : 2020-04-24 21:26
# @Author : zp.liu

import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


class SSD300:
    def __init__(self, session, isTraining):
        self.session = session
        self.isTraining = isTraining
        self.img_size = [300, 300]
        self.classes_size = 21
        self.background_classes_val = 0
        self.default_box_size = [4, 6, 6, 6, 4, 4]
        self.box_aspect_ratio = [
            [1.0, 1.25, 2.0, 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0],
            [1.0, 1.25, 2.0, 3.0]
        ]
        self.min_box_scale = 0.05
        self.max_box_scale = 0.9
        self.default_box_scale =np.linspace(self.min_box_scale, self.max_box_scale, num = np.amax(self.default_box_size))
        print('##   default_box_scale:' + str(self.default_box_scale))
        self.conv_strides_1 = [1, 1, 1, 1]
        self.conv_strides_2 = [1, 2, 2, 1]
        self.conv_strides_3 = [1, 3, 3, 1]
        self.pool_size = [1, 2, 2, 1]
        self.pool_strides = [1, 2, 2, 1]
        self.conv_bn_decay = 0.99999
        self.conv_bn_epsilon = 0.00001
        self.jaccard_value = 0.6
        self.generate_graph()

    def generate_graph(self):
        pass

    def convolution(self, input, shape, padding, strides, name, batch_normalization = True):
        with tf.variable_scope(name):
            weight = tf.get_variable(initializer = tf.truncated_normal(shape, 0, 1), dtype = tf.float32, name = name + '_weight')
            bias = tf.get_variable(initializer = tf.truncated_normal(shape[-1:], 0, 1),dtype = tf.float32, name = name + '_bias')
            result = tf.nn.conv2d(input, shape, padding, strides, name = name + '_conv')
            result = tf.nn.bias_add(result, bias)
            if batch_normalization:
                result = self.batch_normalization(name = name + '_bn')
            result = tf.nn.relu(result, name = name + '_relu')
            return result

    def batch_normalization(self, name):
        pass





if __name__ == '__main__':
    ssd = SSD300(None, None)