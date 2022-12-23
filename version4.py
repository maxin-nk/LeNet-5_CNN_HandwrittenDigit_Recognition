# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
# ======================================================================增加神经网络深度
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


mnist = input_data.read_data_sets("mnist/", one_hot=True)
x_data = tf.placeholder(tf.float32, [None, 28*28])
y_data = tf.placeholder(tf.float32, [None, 10])

weight1 = tf.Variable(tf.zeros([784, 256]))
bias1 = tf.Variable(tf.zeros([256]))
y_model1 = tf.nn.sigmoid(tf.matmul(x_data, weight1) + bias1)

weight2 = tf.Variable(tf.zeros([256, 10]))
bias2 = tf.Variable(tf.zeros([10]))
y_model = tf.nn.softmax(tf.matmul(y_model1, weight2) + bias2)

# 损失函数：交叉熵
loss = -tf.reduce_sum(y_data*tf.log(y_model))

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x_data: batch_xs, y_data: batch_ys})

        if i % 50 == 0:
            correct_prediction = tf.equal(tf.argmax(y_model, axis=1), tf.argmax(y_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))





