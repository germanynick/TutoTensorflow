# CODE: https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py

import argparse
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  init = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(init)


def bias_variable(shape):
  init = tf.truncated_normal(shape=shape, stddev=0.1)
  return tf.Variable(init)


def deepnn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer. 1 -> 32 features
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # First Pooling layer
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer. 32 -> 64 features
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second Pooling layer
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Flat Pooling layer
  with tf.name_scope('flat_pool2'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  # Fully connected layer 1 -- after 2 round of downsampling,
  # Our 28x28 image is down to 7x7x64 feature maps
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout
  with tf.name_scope('dropout'):
    keep_prod = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prod)

  # Output layer
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv, keep_prod


def main(_):
  # DATA
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # input placeholder
  x = tf.placeholder(tf.float32, [None, 784], name='actual_input')

  # output placeholder
  y_ = tf.placeholder(tf.float32, [None, 10], name='actual_output')

  # Build graph for deep net
  y_conv, keep_prod = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('Adam_optimiter'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

  graph_location = 'mnist_deep_graph'
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
      a_input, a_output = mnist.train.next_batch(50)
      if i % 1000 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: a_input, y_: a_output, keep_prod: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: a_input, y_: a_output, keep_prod: 0.7})

    print('test accuracy %g' % accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prod: 1.0}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='MNIST_DATA/',
                      help='Directory for storing input data')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
