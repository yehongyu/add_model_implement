import tensorflow as tf

# Create a graph.
g = tf.Graph()

# Establish the graph as the default graph.
with g.as_default():
    # Use operations to create graph struction.
    x = tf.constant(8, name='x_const')
    y = tf.constant(5, name='y_const')
    sum = tf.add(x, y, name='x_y_sum')

    z = tf.constant(4, name='z_const')
    new_sum = tf.add(sum, z, name="x_y_z_sum")

    die1 = tf.Variable(
        tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32)
    )
    die2 = tf.Variable(
        tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32)
    )
    dice_sum = tf.add(die1, die2)
    result = tf.concat(values=[die1, die2, dice_sum], axis=1)
    # Create a session to run the default graph.
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sum.eval())
        print(new_sum.eval())
        print(result.eval())

