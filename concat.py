import tensorflow as tf
import numpy as np

t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
t3 = tf.concat([t1, t2], axis=-1)

##sess = tf.Session()
##print(sess.run(t3))
##sess.close()

c = np.random.random([10, 1])
d = tf.nn.embedding_lookup(c, [1, 3])

##with tf.Session() as sess:
##    sess.run(tf.initialize_all_variables())
##    print sess.run(d)
##    print c

a = tf.constant([[1,2,3],[3,4,5]]) # shape (2,3)
b = tf.constant([[7,8,9],[10,11,12]]) # shape (2,3)
ab = tf.stack([a,b], axis=1) # shape (2,2,3)

##with tf.Session() as sess:
##    sess.run(tf.initialize_all_variables())
##    print sess.run(ab)

##with tf.Session() as sess:
##    writer = tf.summary.FileWriter('.', sess.graph)

a1 = tf.get_variable('a', shape=[2, 5])
b1 = a1

a_drop = tf.nn.dropout(a1, 0.5)
##with tf.Session() as sess:
##    sess.run(tf.initialize_all_variables())
##    print(sess.run(b1))
##    print(sess.run(a_drop))

c = tf.constant(value=1)
print(c.graph)
print(tf.get_default_graph())

g =tf.Graph()
print('g', g)
with g.as_default():
    d = tf.constant(value=2)
    print(d.graph)

g2 = tf.Graph()
print('g2:', g2)
g2.as_default()
e = tf.constant(value=15)
print(e.graph)








