#coding=utf-8
import tensorflow as tf
import sys
import struct

SAVE_PATH = 'data/dataset.tfrecords'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def write_tfr_with_writer(num=10):
    writer = tf.python_io.TFRecordWriter(SAVE_PATH)
    for i in range(num):
        label = i % 3
        f0 = i * 100 + 5
        f1 = i * 111 + 15
        example = tf.train.Example(features=tf.train.Features(feature={
        'f0': _float_feature(f0),
        'f1': _float_feature(f1),
        'label': _int64_feature(label),
        }))
        writer.write(example.SerializeToString())
    writer.close()

def write_tfr(num=10):
    writer = open(SAVE_PATH, 'wb')
    for i in range(num):
        label = i % 3
        f0 = i * 100 + 5
        f1 = i * 111 + 15
        example = tf.train.Example(features=tf.train.Features(feature={
            'f0': _float_feature(f0),
            'f1': _float_feature(f1),
            'label': _int64_feature(label),
        }))
        data = example.SerializeToString()
        writer.write(struct.pack('<Q', len(data)))
        writer.write(struct.pack('%ds' % len(data), data))
    writer.close()


def load_data_test1(num):
    def get_next(record_iterator):
        string_record = next(record_iterator)
        example = tf.train.Example()
        example.ParseFromString(string_record)
        return example
    record_iterator = tf.python_io.tf_record_iterator(SAVE_PATH)
    while True:
        try:
            example = get_next(record_iterator)
            label = example.features.feature['label']
            print(label.int64_list.value, example.features.feature['f0'].float_list.value, example.features.feature['f1'].float_list.value)
        except StopIteration:
            break

def load_data_test2(num):
    tf.enable_eager_execution()
    raw_dataset = tf.data.TFRecordDataset([SAVE_PATH])
    feature_description = {
        'f0': tf.FixedLenFeature([1], tf.float32),
        'f1': tf.FixedLenFeature([1], tf.float32),
        'label': tf.FixedLenFeature([1], tf.int64),
    }
    for i in raw_dataset.take(1):
        item = i
        example = tf.parse_single_example(item, feature_description)
        print(example) # map:<string, tensor>
        print(example['label'], example['f0'], example['f1'])

if __name__ == '__main__':
    num = 4
    write_tfr_with_writer(num)
    load_data_test1(num)
    '''
    f0_batch, f1_batch, label_batch = load_data_with_reader()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        f0_val, f1_val, label_val = sess.run([f0_batch, f1_batch, label_batch])
        print(f0_val)
        print(f0_val)
        print(f0_val)
    '''


