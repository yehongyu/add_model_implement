#coding=utf-8

import tensorflow as tf

my_data = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7]
]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

sess = tf.Session()

'''
while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break
'''
r = tf.random_normal([10, 3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

'''
sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break
'''


features = {
    'sales': [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']
}
department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening']
)
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]
inputs = tf.feature_column.input_layer(features, columns)

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
print(sess.run(inputs))

