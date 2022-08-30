import logging
import time

import numpy as np
# import matplotlib.pyplot as plt

# import tensorflow_datasets as tfds
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# ## Download the Dataset

# ## Masking

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])


def scaled_dot_product_attention(q):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """
  k=q
  v=q
  mask=None
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

# As the softmax normalization is done along the dimension for keys, the attention values decide the amount of importance given to the keys for each query.
#
# The output represents the multiplication of the attention weights and the V (value) vector. This ensures that the tokens you want to focus on are kept as-is and the irrelevant tokens are flushed out.



np.set_printoptions(suppress=True)





# + [markdown] id="kmzGPEy64qmA"
# ## Multi-head attention

# + [markdown] id="fz5BMC8Kaoqo"
# <img src="https://www.tensorflow.org/images/tutorials/transformer/multi_head_attention.png" width="500" alt="multi-head attention">
#
#
# Multi-head attention consists of four parts:
# *    Linear layers.
# *    Scaled dot-product attention.
# *    Final linear layer.

# + [markdown] id="JPmbr6F1C-v_"
# Each multi-head attention block gets three inputs; Q (query), K (key), V (value). These are put through linear (Dense) layers before the multi-head attention function.
#
# In the diagram above `(K,Q,V)` are passed through sepearte linear (`Dense`) layers for each attention head. For simplicity/efficiency the code below implements this using a single dense layer with `num_heads` times as many outputs. The output is rearranged to a shape of `(batch, num_heads, ...)` before applying the attention function.
#
# The `scaled_dot_product_attention` function defined above is applied in a single call, broadcasted for efficiency. An appropriate mask must be used in the attention step.  The attention output for each head is then concatenated (using `tf.transpose`, and `tf.reshape`) and put through a final `Dense` layer.
#
# Instead of one single attention head, Q, K, and V are split into multiple heads because it allows the model to jointly attend to information from different representation subspaces at different positions. After the split each head has a reduced dimensionality, so the total computation cost is the same as a single head attention with full dimensionality.

# + [markdown] id="0D8FJue5lDyZ"
# Create a `MultiHeadAttention` layer to try out. At each location in the sequence, `y`, the `MultiHeadAttention` runs all 8 attention heads across all other locations in the sequence, returning a new vector of the same length at each location.

# + id="Hu94p-_-2_BX"
# temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((8, 2000, 512))  # (batch_size, encoder_sequence, d_model)


@tf.function(jit_compile=True)
def jit_temp_mha(v):
#   out, _ = temp_mha(y, k=y, q=y, mask=None)
  out = scaled_dot_product_attention(y)
  return out

def non_jit_temp_mha(v):
  out, _ = scaled_dot_product_attention(y)
  return out

# out, attn = temp_mha(y, k=y, q=y, mask=None)
out = jit_temp_mha(y)
# non_jit_out = non_jit_temp_mha(y,y,y)
# np.testing.assert_allclose(out, non_jit_out, atol=1e-5)


print("@@@ Finished @@@")