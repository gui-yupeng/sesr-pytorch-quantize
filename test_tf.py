
import tensorflow as tf
import numpy as np

a = np.array([[1, 2],
              [5, 3],
              [2, 6]])

b = tf.compat.v1.Variable(a)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(b))
    print('************')
    axes = sess.run( tf.compat.v1.range(tf.rank(b) - 1) )
    print(axes)
