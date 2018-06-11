# Helper functions for training the dilated convolutional neural network.
# Dependencies: Python3, Numpy, TensorFlow
# 
# Dilated convolutions are passed over the input data, alternating with max-pooling layers.
# The down-sampled features are then flattened and connected via dense layers to the output nodes.
# Softmax is applied to the output nodes so that each node represents the classification probability.
# Cross-entropy is used as the loss function, and the model is optimized using stochastic gradient descent.

import tensorflow as tf
import csv
import numpy as np

# parses one column from the csv file
# header is the number of rows at the beginning that don't contain data
def csv_parser(file, column, header, dtype):
    ret = []
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if header > 0:
                header -= 1
            elif len(row) <= column:
                break
            else:
                ret.append(float(row[column]))
    return np.array(ret, dtype=dtype)

# Function for creating estimator model
def dilated_cnn_model_pam4(features, labels, mode):
    num_classes = 4
    samples_per_label = 16
    input_layer = tf.reshape(features["x"], [-1, samples_per_label, 1])

    # dilated convolutional layer
    conv1 = tf.layers.conv1d(
        inputs=input_layer, 
        filters=4, 
        kernel_size=8,
        padding='same',
        dilation_rate=2,
        activation=tf.nn.relu,
        name="dilated_conv_1")

    # max-pooling layer
    pool1 = tf.layers.max_pooling1d(
        inputs=conv1,
        pool_size=2,
        strides=2,
        name="max_pool_1")

    # dilated convolutional layer
    conv2_filters = 16
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=conv2_filters, 
        kernel_size=4,
        padding='same',
        dilation_rate=2,
        activation=tf.nn.relu,
        name="dilated_conv_2")

    # max-pooling layer
    pool2 = tf.layers.max_pooling1d(
        inputs=conv2, 
        pool_size=2,
        strides=2,
        name="max_pool_2")

    # write kernel for TensorBoard visualization
    kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'dilated_conv_2/kernel')[0]
    kernel_txt = tf.Print(tf.as_string(kernel), [tf.as_string(kernel)], message='kernel_txt',name='kernel_txt')
    tf.summary.text('kernel_txt', kernel_txt)

    # flatten features
    pool2_flat = tf.reshape(pool2, [-1, 4*conv2_filters])
    dense = tf.layers.dense(inputs=pool2_flat, units=16, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=num_classes)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Same as dilated_cnn_model_pam4 except there are only two output nodes
def dilated_cnn_model_binary(features, labels, mode):
    num_classes = 2
    samples_per_label = 16
    input_layer = tf.reshape(features["x"], [-1, samples_per_label, 1])

    conv1 = tf.layers.conv1d(
        inputs=input_layer, 
        filters=4, 
        kernel_size=8,
        padding='same',
        dilation_rate=2,
        activation=tf.nn.relu,
        name="dilated_conv_1")

    pool1 = tf.layers.max_pooling1d(
        inputs=conv1,
        pool_size=2,
        strides=2,
        name="max_pool_1")

    conv2_filters = 16
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=conv2_filters, 
        kernel_size=4,
        padding='same',
        dilation_rate=2,
        activation=tf.nn.relu,
        name="dilated_conv_2")

    pool2 = tf.layers.max_pooling1d(
        inputs=conv2, 
        pool_size=2,
        strides=2,
        name="max_pool_2")

    kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'dilated_conv_2/kernel')[0]
    kernel_txt = tf.Print(tf.as_string(kernel), [tf.as_string(kernel)], message='kernel_txt',name='kernel_txt')
    tf.summary.text('kernel_txt', kernel_txt)

    pool2_flat = tf.reshape(pool2, [-1, 4*conv2_filters])
    dense = tf.layers.dense(inputs=pool2_flat, units=16, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=num_classes)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

