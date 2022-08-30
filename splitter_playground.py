import tensorflow as tf
import numpy as np


@tf.function(experimental_compile=True)
def test_splitter(A,B,C):
    return  A@B@C

A = tf.random.uniform((10000, 100))
B = tf.random.uniform((100, 10000))
C = tf.random.uniform((10000, 10))

z = test_splitter(A,B,C)
print(z)


