#tensors has gpu supports and is optimized for fast computation
#They are immutable (can't update only can create new ones)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# x = tf.constant(4, shape=(1,1), dtype=tf.float32)
# print(x)
# y= tf.random.normal((3,3), mean=0, stddev=1)
# print(y)

# z = tf.range(10)
# print(z)

# # cast
# a = tf.cast(z, dtype=tf.float16)
# print(a)

#element wise operations 
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

z = tf.add(x,y)
print(z)