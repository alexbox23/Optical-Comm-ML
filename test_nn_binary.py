# Code for testing the neural network on binary data.

import tensorflow as tf
import csv
import numpy as np

import dilated_neural_net

tf.logging.set_verbosity(tf.logging.INFO)

# parse the csv files
data = dilated_neural_net.csv_parser('data/data_Binary_NRZ_RX(small).csv', 1, 7, np.float32)
labels = dilated_neural_net.csv_parser('data/labels_Binary_NRZ_TX.csv', 0, 0, np.int32)
samples_per_label = 16
data = np.reshape(data, [-1, samples_per_label])

training_portion = 1/2
training_size = int(len(data)*training_portion)
training_set = data[:training_size]
test_set = data[training_size:]
training_labels = labels[:training_size]
test_labels = labels[training_size:]

# initialize the neural network model
signal_classifier = tf.estimator.Estimator(
    model_fn=dilated_neural_net.dilated_cnn_model_binary,
    model_dir="/tmp/dilated_cnn_model_binary",
    config=tf.estimator.RunConfig().replace(save_summary_steps=10))

tensors_to_log = {"probabilites": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# train the neural network
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": training_set},
    y=training_labels,
    batch_size=64,
    num_epochs=None,
    shuffle=True)

signal_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

# run the trained model on the test data
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_set},
    y=test_labels,
    num_epochs=1,
    shuffle=False)

eval_results = signal_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

