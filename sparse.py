#!/usr/bin/python3

import tensorflow as tf;

def Dense2Sparse():
  dense = tf.keras.Input((None, None, None)); # dense.shape = (batch, num_heads, query_length, key_length)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length or 1, key_length)
  reshaped_mask = tf.keras.layers.Lambda(lambda x: tf.cond(tf.math.not_equal(tf.shape(x[0])[2], tf.shape(x[1])[2]), lambda: tf.tile(x[0], [1,1,tf.shape(x[1])[2],1,]), lambda: x[0]))([mask, dense]); # mask.shape = (batch, 1, query_length, key_length)
  reshaped_mask = tf.keras.layers.Lambda(lambda x: tf.tile(x[0],[1,tf.shape(x[1])[1],1,1]))([reshaped_mask, dense]); # mask.shape = (batch, num_heads, query_length, key_length)
  indices = tf.keras.layers.Lambda(lambda x: tf.where(tf.cast(x, dtype = tf.int32)))(reshaped_mask); # indices.shape = (num non zero values, 4)
  values = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([dense, indices]); # values.shape = (num non zero values)
  sparse = tf.keras.layers.Lambda(lambda x: tf.sparse.SparseTensor(x[0], values = x[1], dense_shape = tf.cast(tf.shape(x[2]), dtype = tf.int64)))([indices, values, dense]);
  return tf.keras.Model(inputs = (dense, mask), outputs = sparse);

class MaskedDenseMatMul(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(MaskedDenseMatMul, self).__init__(**kwargs);
  def call(self, inputs):
    a = inputs[0]; # a.shape = (batch, heads, query_length, key_dim // heads)
    b = inputs[1]; # b.shape = (batch, heads, key_length, key_dim // heads)
    mask = inputs[2]; # mask.shape = (batch, 1, query_length, key_length)
    reshaped_a = tf.reshape(a, (-1, tf.shape(a)[-2], tf.shape(a)[-1])); # reshaped_a.shape = (batch * heads, query_length, key_dim // heads)
    reshaped_b = tf.reshape(b, (-1, tf.shape(b)[-2], tf.shape(b)[-1])); # reshaped_b.shape = (batch * heads, key_length, key_dim // heads)
    tiled_mask = tf.tile(mask, (1,tf.shape(b)[1],1,1)); # mask.shape = (batch, heads, query_length, key_length)
    reshaped_mask = tf.reshape(tiled_mask, (-1, tf.shape(tiled_mask)[-2], tf.shape(tiled_mask)[-1])); # reshaped_mask.shape = (batch * heads, query_length, key_length)
    def stop_cond(i, a, b, mask, results):
      return tf.math.less(i, tf.shape(a)[0]);
    def row_slice(i, a, b, mask, results):
      a_row = a[i:i+1,...]; # a_row.shape = (1, key_dim // heads);
      mask_row = tf.expand_dims(mask[i,...], axis = -1); # mask_row.shape = (key_length, 1);
      tiled_mask_row = tf.tile(mask_row, (1, tf.shape(b)[-1])); # tiled_mask_row.shape = (key_length, key_dim // heads)
      indices = tf.where(tf.cast(tiled_mask_row, dtype = tf.int32)); # indices.shape = (none-zero num, 2)
      # NOTE: b.shape = (key_length, key_dim // heads)
      values = tf.gather_nd(b, indices);
      masked_b = tf.sparse.SparseTensor(indices, values = values, dense_shape = tf.cast(tf.shape(b), dtype = tf.int64)); # masked_b.shape = (key_length, key_dim // heads)
      qk = tf.sparse.sparse_dense_matmul(a_row, tf.sparse.transpose(masked_b)); # qk.shape = (1, key_length)
      results = tf.concat([results, qk], axis = 0); # results.shape = (n, key_length)
      i += 1;
      return i, a, b, mask, results;
    def dot(x):
      a = x[0]; # a.shape = (query_length, key_dim // heads)
      b = x[1]; # b.shape = (key_length, key_dim // heads)
      mask = x[2]; # mask.shape = (query_length, key_length)
      i, a, b, mask, results = tf.while_loop(cond = stop_cond, body = row_slice,
                                             loop_vars = [tf.constant(0), a, b, mask, tf.zeros((0, tf.shape(b)[0]), dtype = tf.float32)],
                                             shape_invariants = [tf.TensorShape([]), a.shape, b.shape, mask.shape, tf.TensorShape([None, None])],);
      return results; # results.shape = (query_length. key_length)
    results = tf.map_fn(dot, (reshaped_a, reshaped_b, reshaped_mask), dtype = tf.float32); # results.shape = (batch * heads, query_length, key_length)
    results = tf.reshape(results, (tf.shape(a)[0], tf.shape(a)[1], tf.shape(results)[-2], tf.shape(results)[-1])); # results.shape = (batch, heads, query_length, key_length)
    return results;

class SparseDenseMatMul(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(SparseDenseMatMul, self).__init__(**kwargs);
  def call(self, inputs):
    a = inputs[0]; # a.shape = (batch, heads, query_length, key_length)
    b = tf.cast(inputs[1], dtype = tf.float32); # b.shape = (batch, heads, key_length, value_dim)
    reshaped_a = tf.sparse.reshape(a, (-1, tf.shape(a)[-2], tf.shape(a)[-1])); # reshaped_a.shape = (batch * heads, query_length, key_length)
    reshaped_b = tf.reshape(b, (-1, tf.shape(b)[-2], tf.shape(b)[-1])); # reshaped_b.shape = (batch * heads, key_length, value_dim)
    def dot(x):
      a = x[0];
      b = x[1];
      c = tf.sparse.sparse_dense_matmul(a,b);
      return c; # c.shape = (query_length, value_dim)
    results = tf.map_fn(dot, (reshaped_a, reshaped_b), dtype = tf.float32);
    results = tf.reshape(results, (tf.shape(a)[0], tf.shape(a)[1], tf.shape(results)[-2], tf.shape(results)[-1]));
    return results;

class SparseDropout(tf.keras.layers.Layer):
  def __init__(self, drop_rate = 0.2, **kwargs):
    self.drop_rate = drop_rate;
    super(SparseDropout, self).__init__(**kwargs);
  def call(self, inputs):
    keep_tensor = (1 - self.drop_rate) + tf.random.uniform(tf.shape(inputs)); # keep_tensor.shape = (batch, num_heads, seq_length, dim)
    keep_mask = tf.cast(tf.math.floor(keep_tensor), dtype = tf.bool);
    results = tf.sparse.retain(inputs, keep_mask) / (1 - self.drop_rate);
    return results;
  def get_config(self):
    config = super(SparseDropout, self).get_config();
    config['drop_rate'] = self.embed_dim;
    return config;
