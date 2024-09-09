#tensors has gpu supports and is optimized for fast computation
#They are immutable (can't update only can create new ones)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

x = tf.constant(4, shape=(1,1), dtype=tf.float32)
print(x)