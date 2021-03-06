import tensorflow as tf


'''
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(y, {x:[[1, 2, 3], [4, 5, 6]]}))
'''

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)

y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))


loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(i, loss_value)


