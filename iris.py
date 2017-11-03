from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import tensorflow as tf
import numpy as np

IRIS_TRAINING = 'iris_data/iris_training.csv'
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_data/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
  def check_download(file_name, file_url):
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(file_name):
      dirname = os.path.dirname(file_name)
      if not os.path.exists(dirname):
        os.makedirs(dirname)
      raw = urllib.request.urlopen(file_url).read()
      with open(file_name, 'wb') as f:
        f.write(raw)

  def load_data(file_name):
    return tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=file_name,
        target_dtype=np.int,
        features_dtype=np.float32)

  check_download(IRIS_TRAINING, IRIS_TRAINING_URL)
  check_download(IRIS_TEST, IRIS_TEST_URL)

  train_sets = load_data(IRIS_TRAINING)
  test_sets = load_data(IRIS_TEST)

  # feature columns
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units
  classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir='/tmp/iris_model')

  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(
      train_sets.data)}, y=np.array(train_sets.target), num_epochs=None, shuffle=True)
  
  # Train model
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(test_sets.data)}, y=np.array(test_sets.target), num_epochs=1, shuffle=False)

  # Evaluate accuracy
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
  print("\nTest accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  new_sample = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float)

  predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": new_sample}, num_epochs=1, shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predict_classes = [p["classes"] for p in predictions]

  print("New sample, Class Predictions: {}\n".format(predict_classes) )


if __name__ == "__main__":
  main()
  print('Done')
