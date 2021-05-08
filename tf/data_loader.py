# coding=utf-8

import tensorflow as tf

random_data = tf.random_uniform([4, 10])
dataset1 = tf.data.Dataset.from_tensor_slices(
    random_data
)
print(dataset1.output_types)
print(dataset1.output_shapes)

dataset2 = tf.data.Dataset.from_tensor_slices(
    {"a": tf.random_uniform([4]),
     "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)}
)
print(dataset2.output_types)
print(dataset2.output_shapes)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)
print(dataset3.output_shapes)

## 单词迭代器，一次迭代，不需要显式初始化，不支持参数化
dataset = tf.data.Dataset.range(10)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
'''
with tf.Session() as sess:
    for i in range(10):
        value = sess.run(next_element)
        assert i == value
        print(i, value)
'''

## 可初始化的迭代器：需要显式运行iterator.initializer操作。
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

'''
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess.run(next_element)
        assert i == value
        print(i, value)

    sess.run(iterator.initializer, feed_dict={max_value: 100})
    for i in range(100):
        value = sess.run(next_element)
        assert i == value
'''

## 可重新初始化迭代器：通过多个不同的dataset对象进行初始化；
## 两个数据集结构一样，比如训练集和验证集，可使用相同结构的迭代器来初始化；
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)
)
validation_dataset = tf.data.Dataset.range(50)

iterator = tf.data.Iterator.from_structure(
    training_dataset.output_types,
    training_dataset.output_shapes
)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

'''
with tf.Session() as sess:
    for _ in range(1):
        sess.run(training_init_op)
        for i in range(100):
            val = sess.run(next_element)
            print i, val
        sess.run(validation_init_op)
        for i in range(50):
            val = sess.run(next_element)
            ##print i, val
'''

## 可馈送迭代器：与tf.placeholder一起使用；
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)
).repeat()
validation_dataset = tf.data.Dataset.range(50)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes
)
next_element = iterator.get_next()

training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

'''
with tf.Session() as sess:
    sess.run(validation_iterator.initializer)
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    print(training_handle)
    print(validation_handle)
    for i in range(200):
        val = sess.run(next_element, feed_dict={handle: training_handle})
        print(i, val)

    for i in range(50):
        val = sess.run(next_element, feed_dict={handle: validation_handle})
        ##print(i, val)

'''

def dataset_input_fn():
    filenames = ['', '']
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
            "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.image.decode_jpeg(parsed["image_data"])
        image = tf.reshape(image, [299, 299, 1])
        label = tf.cast(parsed["label"], tf.int32)
        feature = {"image_data": image, "date_time": parsed["date_time"]}
        return feature, label

    dateeset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    num_epochs = 10
    dataset = dataset.repeat(num_epochs)

    return dataset







