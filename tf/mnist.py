# coding=utf-8

import numpy as np
import tensorflow as tf

print("tf version:", tf.VERSION)
print("tf keras version:", tf.keras.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(type(x_train), x_train.shape)
print(type(y_train), y_train.shape)
print(type(x_test), x_test.shape)
print(type(y_test), y_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

print 'evaluate'
result = model.evaluate(x_test, y_test)
print type(result), result
print x_test[0]
predict_res = model.predict(x_test, batch_size=1)
print predict_res.shape
print predict_res[0]
print np.argmax(predict_res[0])
