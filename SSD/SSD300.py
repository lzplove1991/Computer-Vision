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
        self.input = tf.placeholder(shape=[None, self.img_size[0], self.img_size[1], 3], dtype = tf.float32, name = 'input_image')

        self.conv_1_1 = self.convolution(self.input, [3, 3, 3, 32], strides = self.conv_strides_1, name = 'conv_1_1')
        self.conv_1_2 = self.convolution(self.conv_1_1, [3, 3, 32, 32], strides = self.conv_strides_1, name = 'conv_1_2')
        self.conv_1_3 = tf.nn.avg_pool(self.conv_1_2, self.pool_size, self.pool_strides, padding = 'SAME',name = 'pool_1_2')



    def convolution(self, input, shape, strides, padding = 'SAME', batch_normalization = True, name = 'convolution_layers'):
        with tf.variable_scope(name):
            weight = tf.get_variable(initializer = tf.truncated_normal(shape, 0, 1), dtype = tf.float32, name = name + '_weight')
            bias = tf.get_variable(initializer = tf.truncated_normal(shape[-1:], 0, 1),dtype = tf.float32, name = name + '_bias')
            result = tf.nn.conv2d(input, weight, strides, padding , name = name + '_conv')
            result = tf.nn.bias_add(result, bias)
            if batch_normalization:
                result = self.batch_normalization(result, name = name + '_bn')
            result = tf.nn.relu(result, name = name + '_relu')
            return result

    def batch_normalization(self, input, name):
        with tf.variable_scope(name):
            bn_input_shape = input.get_shape()
            moving_mean = tf.get_variable(name + '_neam', bn_input_shape[-1:], initializer = tf.zeros_initializer, trainable = False)
            moving_variance = tf.get_variable(name + '_variance', bn_input_shape[-1:], initializer = tf.ones_initializer, trainable = False)
            def mean_var_with_update():
                mean, variance = tf.nn.moments(input, list(range(len(bn_input_shape) - 1)), name = name + '_moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, self.conv_bn_decay),
                                              assign_moving_average(moving_variance, variance, self.conv_bn_decay)]):
                    return tf.identity(mean), tf.identity(variance)
            mean, variance = tf.cond(tf.case(True, tf.bool), mean_var_with_update, lambda : (moving_mean, moving_variance))
            beta = tf.get_variable(name + '_beta', bn_input_shape[-1:],initializer = tf.zeros_initializer)
            gamma = tf.get_variable(name + '_gamma', bn_input_shape[-1:], initializer = tf.ones_initializer)
            return tf.nn.batch_normalization(input, mean, variance, beta, gamma, self.conv_bn_epsilon, name + '_bn_opt')

    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))

    def fc(self, input, out_shape, name):
        with tf.variable_scope(name + '_fc'):
            in_shape = 1
            for d in input.get_shape().as_list()[1:]:
                in_shape += 1
            weight = tf.get_variable(initializer = tf.truncated_normal([in_shape, out_shape], 0, 1), dtype = tf.float32, name = name + '_fc_weight')
            bias = tf.get_variable(initializer = tf.truncated_normal([out_shape], 0, 1), dtype = tf.float32, name = name + '_fc_bias')
            result = tf.reshape(input, [-1, in_shape])
            result = tf.nn.xw_plus_b(result, weight, bias, name = name + '_fc_do')
            return result





if __name__ == '__main__':
    ssd = SSD300(None, None)