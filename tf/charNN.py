# coding=utf-8
import collections
import csv

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.01, 'Learning rate.')
tf.app.flags.DEFINE_integer("num_steps", 1000, "Number steps.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size.")
tf.app.flags.DEFINE_integer("display_step", 10, "Display step.")

FLAGS = tf.app.flags.FLAGS

# Network parameters
n_hidden_1 = 320
n_hidden_2 = 160
num_input = 784
num_calsses = 15

# Load data
def read_tsv(input_file, quotechar=None):
    with tf.gfile.open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def load_data(input_file):
    data = read_tsv(input_file)
    features = []
    labels = []
    for d in data:
        features.append([float(v) for v in d[:-1]])
        labels.append(float(d[-1]))
    return np.array(features), np.array(labels)

# Define the neural network
def neural_net(x_dict):
    x = x_dict['features']
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    out_layer = tf.layers.dense(layer_2, num_calsses)
    return out_layer

# Define the model function
def model_fn(features, labels, mode):
    logits = neural_net(features)

    # Prediction
    pred_classes = tf.arg_max(logits, dimension=1)
    pred_probas = tf.nn.softmax(logits)

    # Early return for prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)
    ))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes,
                                             loss=loss_op, train_op=train_op,
                                             eval_metric_ops={"accuracy": acc_op})
    return estim_specs

# Build model
model = tf.estimator.Estimator(model_fn)

# Define input function for training
features, labels = load_data('')
input_fn = tf.estimator.inputs.numpy_input_fn(x={'features': features},
                                              y=labels, batch_size=FLAGS.batch_size,
                                              num_epochs=None, shuffle=True)

# Train model
model.train(input_fn, steps=FLAGS.num_steps)

# Define the input function for evaluating
features, labels = load_data('')
input_fn = tf.estimator.inputs.numpy_input_fn(x={'features': features},
                                              y=labels, batch_size=FLAGS.batch_size,
                                              shuffle=False)
# Use the Estimator evaluate method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])



