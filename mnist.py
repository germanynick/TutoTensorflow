# CODE Github
# https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_softmax.py

from __future__ import absolute_import, division, print_function

import argparse
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):
  #import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Input
  x = tf.placeholder(tf.float32, [None, 784])
  # Weight
  W = tf.Variable(tf.zeros([784, 10]))
  # Bias
  b = tf.Variable(tf.zeros([10]))

  # Predict
  y = tf.matmul(x, W) + b

  # Actual
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Loss  
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  # Use gradient descent to minimize loss
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Session
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='MNIST_data/',
                      help="Directory for storing input data")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
