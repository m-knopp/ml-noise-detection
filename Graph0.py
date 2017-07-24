"""
    This file is part of the bachelor thesis about detecting grinding noise
    It builds the first prototype of a tensorflow graph.
    
    Author:
    Manuel Knopp
    
    Graph info:
    3 conv. layers with pooling followed by 2 densely connected layers.
    kernel size 5x5, channels 32/32/16
    
    Input:
    greyscale image of mel-spectrum size[BATCHSIZE, 128, 32, 1]
    
    Output:
    probability of grinding 0.0 ... 1.0 size[BATCHSIZE, 1]
"""

import tensorflow as tf
import numpy as np

def _weight_variable(shape, name):
    init = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(init)

def _bias_variable(shape, name):
    init = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(init)

def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def _max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def Build_Graph(x):
    #Placeholders
    #x = tf.placeholder(tf.float32, shape=[None, 128, 32, 1])
    #y_ = tf.placeholder(tf.float32, shape=[None, 1])

    x = tf.reshape(x, [-1, 128, 32, 1])

    #Network weights
    with tf.name_scope("Variables"):
        W_c2 = _weight_variable([5, 5, 32, 32], "W_c2")
        W_c3 = _weight_variable([5, 5, 32, 16], "W_c3")

        b_c2 = _bias_variable([32], "b_c2")
        b_c3 = _bias_variable([16], "b_c3")

        W_d1 = _weight_variable([1024, 32], "W_d1")
        W_d2 = _weight_variable([32, 1], "W_d2")

        b_d1 = _bias_variable([32], "b_d1")
        b_d2 = _bias_variable([1], "b_d2")

    #TOTAL Network Size: Conv: [800+800+400] = 2000 byte
    #                    Dense:[32768+32] = 32800 byte

    with tf.name_scope("conv1"):
        #Vars
        W_c1 = _weight_variable([5, 5, 1, 32], "W_c1")
        b_c1 = _bias_variable([32], "b_c1")

        #Convolution
        conv1 = tf.nn.relu(_conv2d(x, W_c1) + b_c1)
        pool1 = _max_pool_2x2(conv1)

    with tf.name_scope("conv2"):
        #Second Convolution
        conv2 = tf.nn.relu(_conv2d(pool1, W_c2) + b_c2)
        pool2 = _max_pool_2x2(conv2)

    with tf.name_scope("conv3"):
        #Last Convolution
        conv3 = tf.nn.relu(_conv2d(pool2, W_c3) + b_c3)
        pool3 = _max_pool_2x2(conv3)

    with tf.name_scope("densely"):
        #Densely Connected Layers
        pool3_flat = tf.reshape(pool3, [-1, 16*4*16])
        #dense1 = tf.layers.dense(pool3_flat, 32, activation=tf.nn.relu)
        #dense2 = tf.layers.dense(dense1, 1, activation=tf.nn.sigmoid)

        dense1 = tf.nn.relu(tf.matmul(pool3_flat, W_d1) + b_d1)
        dense2 = tf.nn.relu(tf.matmul(dense1, W_d2) + b_d2)

    #Readout Layer
    y = tf.reshape(tf.nn.sigmoid(tf.reduce_sum(dense2, 1)), [-1, 1])

    return dense2



