# coding=utf-8

import tensorflow as tf

mymat = tf.Variable([[4, 9], [16, 25]], tf.int32)
rank_of_mymat = tf.rank(mymat)
my_image = tf.zeros([10, 299, 299, 3])
rank_of_myimage = tf.rank(my_image)
print(rank_of_myimage)
print(my_image.shape)

float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
print(float_tensor)
print(float_tensor.eval(session=tf.Session()))


## variabel
my_variable = tf.get_variable("my_var", [1, 2, 3], dtype=tf.float32,
                              initializer=tf.zeros_initializer)
my_variable_from_tensor = tf.get_variable("my_var_from_tensor", dtype=tf.int32,
                              initializer=tf.constant([23, 42]))

print(tf.GraphKeys.GLOBAL_VARIABLES)
print(tf.GraphKeys.TRAINABLE_VARIABLES)
##print(tf.Session().run(tf.report_uninitialized_variables()))

x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

'''
with tf.Session() as sess:
    print(sess.run(y, {x: [1.0, 2.0, 3.0]}))
    print(sess.run(y, {x: [8, 6, 5]}))
    print(sess.run(y, {x: 37.0}))
'''

g = tf.get_default_graph()
print(g.get_operations())

g_1 = tf.Graph()
with g_1.as_default():
    c = tf.constant("Node in g_1")
    print(c)
    sess_1 = tf.Session()
    print(sess_1.run(c))

assert tf.get_default_graph() is g

g_2 = tf.Graph()
with g_2.as_default():
    d = tf.constant("Node in g_2")
    print(d)
    sess_2 = tf.Session(graph=g_2)
    print(sess_2.run(d))

assert tf.get_default_graph() is g

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2
