# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


mnist = input_data.read_data_sets("mnist/", one_hot=True)

x_data = tf.placeholder("float", [None, 28*28])
y_data = tf.placeholder("float", [None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.ones(shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


def max_poop_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x_data, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_poop_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_poop_2x2(h_conv2)

W_fc1 = weight_variable([4*4*64, 1024])
b_fc1 = bias_variable([1024])

h_poop2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_poop2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_data*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 5 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x_data: batch[0], y_data: batch[1], keep_prob: 1.0})
        print("step:%s , train accuracy %s" % (i, train_accuracy))

    sess.run(train_step, feed_dict={x_data: batch[0], y_data: batch[1], keep_prob: 1.0})
