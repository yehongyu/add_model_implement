# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import numpy as np
import os
import tensorflow as tf

from mlx_studio.modelhub import register_model
from mlx_studio.modelhub.modelhub import ModelType
from mlx_studio.storage import hdfs
from mlx_studio.context import JOB_CONTEXT


def load_data(hdfs_path, local_path='mnist.npz'):
    if not os.path.exists(local_path):
        hdfs.download(local_path, hdfs_path)

    with np.load(local_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


def train(model_path, train_paths, log_dir):
    (x_train, y_train), (x_test, y_test) = load_data(hdfs_path=train_paths)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])
    model.evaluate(x_test, y_test)
    model.save(model_path, save_format='tf')
    # TODO: use hook to auto register
    path_saved_model =  os.path.join(model_path, 'saved_model.pb')
    register_model(path_saved_model, model_type=ModelType.Tensorflow)


def main():
    parser = argparse.ArgumentParser()
    model_path = JOB_CONTEXT.model_path
    train_path = JOB_CONTEXT.train_path or 'hdfs://haruna/user/yangrun/tests/minist/mnist.npz'
    log_dir = JOB_CONTEXT.summary_dir
    parser.add_argument('--model_path', type=str,
                        default=model_path, metavar='S',
                        help='hdfs:// or file:/// URL to the MNIST model export '
                             '(default: %s)' % model_path)
    parser.add_argument('--train_paths', type=str,
                        default=train_path, metavar='S',
                        help='hdfs:// URL to the MNIST data, (default: %s)' % train_path)
    parser.add_argument('--log_dir', type=str,
                        default=log_dir, metavar='S',
                        help='hdfs:// URL to the tf events dir, (default: %s)' % log_dir)
    args = parser.parse_args()
    train(args.model_path, args.train_paths, args.log_dir)


if __name__ == '__main__':
    main()