# coding=utf-8

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print train_data.shape
print train_labels.shape
print test_data.shape
print test_labels.shape

print(train_data[0])
print(train_labels[0])
print len(train_data[0]), len(train_data[1])


def get_word_index():
    ## 将整数转换回字词
    word_src = imdb.get_word_index()

    word_index = {}
    count = 0
    for k, v in word_src.items():
        ##print k, v
        count += 1
        if count < 10:
            print k, v
        word_index[k] = v+3
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict()
    for key, value in word_index.items():
        reverse_word_index[value] = key
    return word_index, reverse_word_index

def decode_review(text_index, reverse_word_index):
    res_list = []
    for i in text_index:
        res_list.append(reverse_word_index.get(i, '?'))
    return ' '.join(res_list)

word_index, reverse_word_index = get_word_index()
print decode_review(train_data[0], reverse_word_index)

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index['<PAD>'],
    padding='post', maxlen=256
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index['<PAD>'],
    padding='post', maxlen=256
)

print len(train_data[0]), len(train_data[1])
print train_data[0]

## 构建模型
## 设计模型使用多少个层？每一层使用多少个隐藏单元

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

## 选择优化器，损失函数，指标
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

## 训练数据，测试数据
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

## 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)
print results

history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

