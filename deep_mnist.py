# CODE: Github
# https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
sess = tf.InteractiveSession()

# Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Input
x = tf.placeholder(tf.float32, shape=[None, 784])

# Actual_Output
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#------------- SIMPLE ONE LAYER -----------#
# Weight
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Init global variable
sess.run(tf.global_variables_initializer())

# Predict_Output
y = tf.matmul(x, W) + b

# Loss(Actual && Predict Output)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train Model (Gradient Descent)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
for _ in range(1000):
  input, output = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: input, y_: output})

# Test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#------------- SIMPLE ONE LAYER -----------#

#-------------- CNN --------------------#


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  intial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(intial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])

### HIDDEN LAYER 1 (1 x 32) ###
# Weight
W_conv1 = weight_variable([5, 5, 1, 32])
# Bias
b_conv1 = bias_variable([32])

# Pool 1
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

### HIDDEN LAYER 2 (32 x 64) ###
# Weight
W_conv2 = weight_variable([5, 5, 32, 64])
# Bias
b_conv2 = bias_variable([64])

# Pool 2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Flat pool 2
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

### CONNECTED LAYER 1 ###
# Weight
W_fc1 = weight_variable([7 * 7 * 64, 1024])

# Bias
b_fc1 = bias_variable([1024])

# Pool
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

### DROP LAYER ###
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### READOUT LAYER ###
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Final_Predict_Ouput
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# Train Modal
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(10000):
    input, output = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(
          feed_dict={x: input, y_: output, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))

    train_step.run(feed_dict={x: input, y_: output, keep_prob: 0.5})

  test_accuracy = accuracy.eval(
      feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  print('test accuracy %g' % test_accuracy)

#-------------- CNN --------------------#
