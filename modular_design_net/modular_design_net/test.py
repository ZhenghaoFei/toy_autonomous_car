import tensorflow as tf
import numpy

a =  tf.constant([1,10,5,1])
print(a.shape)
b = tf.stack([tf.range(0,4),a], axis=1)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print sess.run(a)
print sess.run(b)
