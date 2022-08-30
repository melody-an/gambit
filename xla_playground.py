import tensorflow as tf
import numpy as np

@tf.function(jit_compile=True)
def test_einsum_reduce_sum(A,B,C):
    return  tf.math.reduce_sum(A@B@C,axis=1)

A = tf.random.uniform((40, 20))
B = tf.random.uniform((20, 30))
C = tf.random.uniform((30, 10))

z = test_einsum_reduce_sum(A,B,C)
print(z)


@tf.function(jit_compile=True)
def test_mco(A,B,C):
    return  A@B@C

A = tf.random.uniform((10000, 100))
B = tf.random.uniform((100, 10000))
C = tf.random.uniform((10000, 10))

z = test_mco(A,B,C)
print(z)


A = tf.random.uniform((40, 20))
B = tf.random.uniform((20, 30))
C = tf.random.uniform((30,20))
D = tf.random.uniform((30,10))
J = tf.random.uniform((20,5))
H = tf.random.uniform((5,10))

@tf.function(jit_compile=True)
def transpose_chain():
  part1 = tf.transpose(D@tf.transpose(J@H)) #10*30
  part2 = tf.transpose(A@B@C@part1)
  return part2

z = transpose_chain()
print(z)