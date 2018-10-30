"""
Tensorflow 101

@author: Aqib Ansari

"""
import numpy as np
import tensorflow as tf

#  constant
const = tf.constant(2.0, name = 'const')

#  variables
b = tf.placeholder(tf.float32, [None, 1], name = 'b' )
c = tf.Variable(1.0, name = 'a')

# operations
d = tf.add(b, c, name = 'd')
e = tf.add(c, const, name = 'e')
a = tf.multiply(d, e, name = 'a')

# initiate variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    a_out = sess.run(a, feed_dict = {b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))
