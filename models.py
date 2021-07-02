#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from blocksparse import BlocksparseTransformer;

class Quantize(tf.keras.layers.Layer):
  def __init__(self, embed_dim = 128, n_embed = 10000, **kwargs):
    self.embed_dim = embed_dim;
    self.n_embed = n_embed;
    super(Quantize, self).__init__(**kwargs);
  def build(self, input_shape):
    # cluster_mean: cluster means which are used as codes for code book
    self.cluster_mean = self.add_weight(shape = (self.embed_dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = True, name = 'cluster_mean');
  def call(self, inputs):
    samples = tf.reshape(inputs, (-1, self.embed_dim,)); # samples.shape = (n_sample, dim)
    # dist = (X - cluster_mean)^2 = X' * X - 2 * X' * Embed + trace(Embed' * Embed),  dist.shape = (n_sample, n_embed), euler distances to cluster_meanding vectors
    dist = tf.math.reduce_sum(samples ** 2, axis = 1, keepdims = True) - 2 * tf.linalg.matmul(samples, self.cluster_mean) + tf.math.reduce_sum(self.cluster_mean ** 2, axis = 0, keepdims = True);
    cluster_index = tf.math.argmin(dist, axis = 1); # cluster_index.shape = (n_sample)
    cluster_index = tf.reshape(cluster_index, tf.shape(inputs)[:-1]); # cluster_index.shape = (batch, h, w)
    quantize = tf.nn.embedding_lookup(tf.transpose(self.cluster_mean), cluster_index); # quantize.shape = (batch, h, w, dim)
    q_loss = tf.math.reduce_mean((quantize - tf.stop_gradient(inputs)) ** 2);
    e_loss = tf.math.reduce_mean((tf.stop_gradient(quantize) - inputs) ** 2);
    loss = q_loss + 0.25 * e_loss;
    outputs = inputs + tf.stop_gradient(quantize - inputs);
    return outputs, cluster_index, loss;
  def get_config(self):
    config = super(Quantize, self).get_config();
    config['embed_dim'] = self.embed_dim;
    config['n_embed'] = self.n_embed;
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

class QuantizeEma(tf.keras.layers.Layer):
  def __init__(self, embed_dim = 128, n_embed = 10000, decay = 0.99, eps = 1e-5, enable_train = True, **kwargs):
    self.embed_dim = embed_dim;
    self.n_embed = n_embed;
    self.decay = decay;
    self.eps = eps;
    self.enable_train = enable_train;
    super(QuantizeEma, self).__init__(**kwargs);
  def build(self, input_shape):
    self.cluster_mean = self.add_weight(shape = (self.embed_dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = True, name = 'cluster_mean');
    self.cluster_size = self.add_weight(shape = (self.n_embed,), dtype = tf.float32, initializer = tf.keras.initializers.Zeros(), trainable = True, name = 'cluster_size');
    self.cluster_sum = self.add_weight(shape = (self.embed_dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = True, name = 'cluster_sum');
    self.cluster_mean.assign(self.cluster_sum);
  def call(self, inputs):
    samples = tf.reshape(inputs, (-1, self.embed_dim,)); # samples.shape = (n_sample, dim)
    # dist = (X - cluster_mean)^2 = X' * X - 2 * X' * Embed + trace(Embed' * Embed),  dist.shape = (n_sample, n_embed), euler distances to cluster_meanding vectors
    dist = tf.math.reduce_sum(samples ** 2, axis = 1, keepdims = True) - 2 * tf.linalg.matmul(samples, self.cluster_mean) + tf.math.reduce_sum(self.cluster_mean ** 2, axis = 0, keepdims = True);
    cluster_index = tf.math.argmin(dist, axis = 1); # cluster_index.shape = (n_sample)
    if self.enable_train:
       # NOTE: code book is updated during forward propagation
      cluster_index_onehot = tf.one_hot(cluster_index, self.n_embed); # cluster_index_onehot.shape = (n_sample, n_embed)
      cluster_size = tf.math.reduce_sum(cluster_index_onehot, axis = 0); # cluster_size.shape = (n_embed)
      cluster_sum = tf.linalg.matmul(samples, cluster_index_onehot, transpose_a = True); # cluster_sum.shape = (dim, n_embed)
      updated_cluster_size = self.cluster_size * self.decay + cluster_size * (1 - self.decay); # updated_cluster_size.shape = (n_embed)
      updated_cluster_sum = self.cluster_sum * self.decay + cluster_sum * (1 - self.decay); # updated_cluster_sum.shape = (dim, n_embed)
      self.cluster_size.assign(updated_cluster_size);
      self.cluster_sum.assign(updated_cluster_sum);
      n_sample = tf.math.reduce_sum(self.cluster_size); # n_sample.shape = ()
      cluster_size = (self.cluster_size + self.eps) * n_sample / (n_sample + self.n_embed * self.eps); # cluster_size.shape = (n_embed)
      cluster_mean = self.cluster_sum / cluster_size; # cluster_mean.shape = (dim, n_embed)
      self.cluster_mean.assign(cluster_mean);
    cluster_index = tf.reshape(cluster_index, tf.shape(inputs)[:-1]); # cluster_index.shape = (batch, h, w)
    quantize = tf.nn.embedding_lookup(tf.transpose(self.cluster_mean), cluster_index); # quantize.shape = (batch, h, w, dim)
    e_loss = tf.math.reduce_mean((inputs - tf.stop_gradient(quantize)) ** 2); # diff.shape = (n_sample,)
    outputs = inputs + tf.stop_gradient(quantize - inputs);
    return outputs, cluster_index, 0.25 * e_loss;
  def set_trainable(self, enable_train = True):
    self.enable_train = enable_train;
  def get_config(self):
    config = super(QuantizeEma, self).get_config();
    config['embed_dim'] = self.embed_dim;
    config['n_embed'] = self.n_embed;
    config['decay'] = self.decay;
    config['eps'] = self.eps;
    config['enable_train'] = self.enable_train;
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

def Encoder(in_channels = 3, out_channels = 128, block_num = 2, res_channels = 32, stride = 4, name = 'encoder'):
  assert stride in [2, 4];
  inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch, height, width, in_channels)
  if stride == 4:
    results = tf.keras.layers.Conv2D(out_channels // 2, (4,4), strides = (2,2), padding = 'same', activation = tf.keras.activations.relu)(inputs);
    results = tf.keras.layers.Conv2D(out_channels, (4,4), strides = (2,2), padding = 'same', activation = tf.keras.activations.relu)(results);
    results = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same')(results);
  elif stride == 2:
    results = tf.keras.layers.Conv2D(out_channels // 2, (4,4), strides = (2,2), padding = 'same', activation = tf.keras.activations.relu)(inputs);
    results = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same')(results);
  else:
    raise Exception('invalid stride option');
  for i in range(block_num):
    short = results;
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(res_channels, (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
    results = tf.keras.layers.Conv2D(out_channels, (1,1), padding = 'same')(results);
    results = tf.keras.layers.Add()([results, short]);
  return tf.keras.Model(inputs = inputs, outputs = results, name = name);

def Decoder(in_channels, out_channels, hidden_channels = 128, block_num = 2, res_channels = 32, strides = 4, name = 'decoder'):
  assert strides in [2, 4];
  inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch, height, width, in_channels)
  results = tf.keras.layers.Conv2D(hidden_channels, (3,3), padding = 'same')(inputs);
  for i in range(block_num):
    short = results;
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(res_channels, (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
    results = tf.keras.layers.Conv2D(hidden_channels, (1,1), padding = 'same')(results);
    results = tf.keras.layers.Add()([results, short]);
  results = tf.keras.layers.ReLU()(results);
  if strides == 4:
    results = tf.keras.layers.Conv2DTranspose(hidden_channels // 2, (4,4), strides = (2,2), padding = 'same', activation = tf.keras.activations.relu)(results);
    results = tf.keras.layers.Conv2DTranspose(out_channels, (4,4), strides = (2,2), padding = 'same', activation = tf.keras.activations.sigmoid)(results);
  elif strides == 2:
    results = tf.keras.layers.Conv2DTranspose(out_channels, (4,4), strides = (2,2), padding = 'same', activation = tf.keras.activations.sigmoid)(results);
  else:
    raise Exception('invalid stride option');
  results = tf.keras.layers.Lambda(lambda x: x - 0.5)(results);
  return tf.keras.Model(inputs = inputs, outputs = results, name = name);

def VQVAE_Encoder(in_channels = 3, hidden_channels = 128, block_num = 2, res_channels = 32, embed_dim = 64, n_embed = 512, quantize_type = 'original', name = 'encoder'):
  quantize_type in ['original', 'ema_update'];
  inputs = tf.keras.Input((None, None, in_channels));
  enc_b = Encoder(in_channels, hidden_channels, block_num, res_channels, 4, name = 'bottom_encoder')(inputs); # enc_b.shape = (batch, h/4, w/4, hidden_channels)
  enc_t = Encoder(hidden_channels, hidden_channels, block_num, res_channels, 2, name = 'top_encoder')(enc_b); # enc_t.shape = (batch, h/8, w/8, hidden_channels)
  results = tf.keras.layers.Conv2D(embed_dim, (1,1))(enc_t); # results.shape = (batch, h/8, w/8, embed_dim)
  if quantize_type == 'original':
    quantized_t, cluster_index_t, loss_t = Quantize(embed_dim, n_embed, name = 'top_quantize')(results); # quantized_t.shape = (batch, h/8, w/8, embed_dim)
  else:
    quantized_t, cluster_index_t, loss_t = QuantizeEma(embed_dim, n_embed, name = 'top_quantize')(results); # quantized_t.shape = (batch, h/8, w/8, embed_dim)
  dec_t = Decoder(embed_dim, embed_dim, hidden_channels, block_num, res_channels, 2)(quantized_t); # dec_t.shape = (batch, h/4, w/4, embed_dim)
  enc_b = tf.keras.layers.Concatenate(axis = -1)([dec_t, enc_b]); # enc_b.shape = (bath, h/4, w/4, embed_dim + hidden_channels)
  results = tf.keras.layers.Conv2D(embed_dim, (1,1))(enc_b); # results.shape = (batch, h/4, w/4, embed_dim)
  if quantize_type == 'original':
    quantized_b, cluster_index_b, loss_b = Quantize(embed_dim, n_embed, name = 'bottom_quantize')(results); # quantized_b.shape = (batch, h/4, w/4, embed_dim)
  else:
    quantized_b, cluster_index_b, loss_b = QuantizeEma(embed_dim, n_embed, name = 'bottom_quantize')(results); # quantized_b.shape = (batch, h/4, w/4, embed_dim)
  return tf.keras.Model(inputs = inputs, outputs = (quantized_t, cluster_index_t, loss_t, quantized_b, cluster_index_b, loss_b), name = name);

def VQVAE_Decoder(in_channels = 3, hidden_channels = 128, block_num = 2, res_channels = 32, embed_dim = 64, name = 'decoder'):
  quantized_t = tf.keras.Input((None, None, embed_dim)); # quantized_t.shape = (batch, h/8, w/8, embed_dim)
  quantized_b = tf.keras.Input((None, None, embed_dim)); # quantized_b.shape = (batch, h/4, w/4, embed_dim)
  results = tf.keras.layers.Conv2DTranspose(embed_dim, (4,4), strides = (2,2), padding = 'same')(quantized_t); # results.shape = (batch, h/4, w/4, embed_dim)
  results = tf.keras.layers.Concatenate(axis = -1)([results, quantized_b]); # results.shape = (batch, h/4, w/4, 2 * embed_dim)
  results = Decoder(2 * embed_dim, in_channels, hidden_channels, block_num, res_channels, 4)(results); # results.shape = (batch, h, w, 3)
  return tf.keras.Model(inputs = (quantized_t, quantized_b), outputs = results, name = name);

class VQVAE_Trainer(tf.keras.Model):
  def __init__(self, in_channels = 3, hidden_channels = 128, block_num = 2, res_channels = 32, embed_dim = 64, n_embed = 512, quantize_type = 'original'):
    super(VQVAE_Trainer, self).__init__();
    self.encoder = VQVAE_Encoder(in_channels, hidden_channels, block_num, res_channels, embed_dim, n_embed, quantize_type);
    self.decoder = VQVAE_Decoder(in_channels, hidden_channels, block_num, res_channels, embed_dim);
  def call(self, inputs):
    quantized_t, cluster_index_t, loss_t, quantized_b, cluster_index_b, loss_b = self.encoder(inputs);
    recon = self.decoder([quantized_t, quantized_b]);
    loss = tf.keras.layers.Add()([loss_t, loss_b]);
    return recon, loss;

def AttentionBlock(embed_dim,):
  inputs = tf.keras.Input((None, None, embed_dim,)); # inputs.shape = (batch, h, w, embed_dim)
  results = tf.keras.layers.LayerNormalization(axis = [1,2,3])(inputs);
  # TODO

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

def Mask(mode = 'all', look_backward_length = None):
  query = tf.keras.Input((None, None, None)); # seq_length.shape = (batch, num_heads, seq_length, dim)
  seq_length = tf.keras.layers.Lambda(lambda x: tf.shape(x)[2])(query); # seq_length.shape = ()
  assert mode in ['all', 'local', 'strided'];
  if mode == 'all':
    results = tf.keras.layers.Lambda(lambda x: tf.linalg.band_part(tf.ones((x,x)), -1, 0))(seq_length);
  elif mode == 'local':
    results = tf.keras.layers.Lambda(lambda x, n: tf.linalg.band_part(tf.ones((x,x)), tf.math.minimum(x-1,n-1), 0), arguments = {'n': look_backward_length})(seq_length);
  elif mode == 'strided':
    y = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(x, dtype = tf.int32),(-1,1)),(1,x)))(seq_length);
    x = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(x, dtype = tf.int32),(1,-1)),(x,1)))(seq_length);
    diff = tf.keras.layers.Lambda(lambda x, n: tf.cast(tf.math.equal(tf.math.floormod(x[0] - x[1], n), 0), dtype = tf.float32), arguments = {'n': look_backward_length})([y,x]);
    results = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x, tf.linalg.band_part(tf.ones_like(x), -1, 0)))(diff);
  else:
    raise Exception('invalid sparse mode');
  results = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(x[0], (1, 1, tf.shape(x[0])[0], tf.shape(x[0])[1])), (tf.shape(x[1])[0],1,1,1)))([results, query]); # results.shape = (batch, 1, seq_length, seq_length)
  return tf.keras.Model(inputs = query, outputs = results);

def FullAttention(key_dim, value_dim, num_heads, drop_rate = 0.5, sparse = None, look_backward_length = 5):
  # NOTE:
  # sparse = None: calculate cross attention, mask must be given
  # sparse = 'all': calculate self attention, mask is lower triangle calculated from shape of query.
  # sparse = 'local': calculate self attention, mask is banded lower triangle calculated from shape of query.
  # sparse = 'strided': calculate self attention, mask is strided lower triangle calculated from shape of query.
  assert sparse in [None, 'all', 'local', 'strided'];
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query.shape = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  # 1) correlation matrix of query and key
  if sparse in [None, 'all']:
    if sparse is None:
      mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length, key_length)
    elif sparse == 'all':
      mask = Mask('all')(query); # mask.shape = (batch, 1, query_length or 1, key_length)
    else:
      raise Exception('unknown sparse type!');
    qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([query, key]); # qk.shape = (batch, heads, query_length, key_length)
  else:
    if sparse == 'local':
      mask = Mask('local', look_backward_length)(query); # mask.shape = (batch, 1, query_length or 1, key_length)
    elif sparse == 'strided':
      mask = Mask('strided', look_backward_length)(query); # mask.shape = (batch, 1, query_length or 1, key_length)
    else:
      raise Exception('unknown sparse type!');
    qk = MaskedDenseMatMul()([query, key, mask]);
  logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads, query_length, key_length)
  logits = Dense2Sparse()([logits, mask]); # logits.shape = (batch, heads, query_length, key_length)
  attention = tf.keras.layers.Lambda(lambda x: tf.sparse.softmax(x))(logits); # attention.shape = (batch, num_heads, query_length, key_length)
  # 2) weighted sum of value elements for each query element
  results = SparseDenseMatMul()([attention, value]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  return tf.keras.Model(inputs = (query, key, value) if sparse is not None else (query, key, value, mask), outputs = results);

def AxialAttention(key_dim, value_dim, num_heads, drop_rate = 0.5, origin_shape = None, axial_dim = 0, sparse = 'all', look_backward_length = 5):
  # NOTE: this attention can only apply to self attention, but cross attention.
  # in other words, query_length = key_length must hold
  # NOTE: leave one dim as seq_length, merge the other dims with heads.
  # for example key.shape = (batch, heads, h, w, c, dim) and axial_dim = -2
  # key.shape becomes (batch, new_heads = heads * h * c, seq_length = w, dim),
  # the self attention matrix become w x w, rather than (h * w * c) x (h * w * c)
  assert type(origin_shape) is list or type(origin_shape) is tuple;
  assert 0 <= axial_dim < len(origin_shape) or -len(origin_shape) <= axial_dim < 0;
  assert sparse is not None;
  query = tf.keras.Input((num_heads, np.prod(origin_shape), key_dim // num_heads)); # query.shape = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, np.prod(origin_shape), key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, np.prod(origin_shape), value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  reshaped_query = tf.keras.layers.Reshape((num_heads, *origin_shape, key_dim // num_heads))(query);
  reshaped_key = tf.keras.layers.Reshape((num_heads, *origin_shape, key_dim // num_heads))(key);
  reshaped_value = tf.keras.layers.Reshape((num_heads, *origin_shape, value_dim // num_heads))(value);
  def get_perm(origin_shape, axial_dim):
    dims = np.arange(2 + len(origin_shape) + 1); # batch x heads x *origin_shape x dim
    chosed_dim = 2 + axial_dim if axial_dim >= 0 else 2 + len(origin_shape) + axial_dim;
    index = dims.tolist().index(chosed_dim);
    dims[index], dims[-2] = dims[-2], dims[index];
    return dims;
  reshaped_query = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(origin_shape, axial_dim)})(reshaped_query);
  reshaped_query = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_query); # query.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  reshaped_key = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(origin_shape, axial_dim)})(reshaped_key);
  reshaped_key = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_key); # key.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  reshaped_value = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(origin_shape, axial_dim)})(reshaped_value);
  shape = tf.keras.layers.Lambda(lambda x: tf.shape(x))(reshaped_value);
  reshaped_value = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_value); # value.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  # 1) correlation matrix of query and key
  if sparse in [None, 'all']:
    mask = Mask('all')(reshaped_query); # mask.shape = (batch, 1, origin_shape[axial_dim], origin_shape[axial_dim])
    qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([reshaped_query, reshaped_key]); # qk.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  else:
    if sparse == 'local':
      mask = Mask('local', look_backward_length)(reshaped_query); # mask.shape = (batch, 1, origin_shape[axial_dim], origin_shape[axial_dim])
    elif sparse == 'strided':
      mask = Mask('strided', look_backward_length)(reshaped_query); # mask.shape = (batch, 1, origin_shape[axial_dim], origin_shape[axial_dim])
    else:
      raise Exception('unknown sparse type!');
    qk = MaskedDenseMatMul()([reshaped_query, reshaped_key, mask]);
  logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  logits = Dense2Sparse()([logits, mask]); # logits.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  attention = tf.keras.layers.Lambda(lambda x: tf.sparse.softmax(x))(logits); # attention.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  # 2) weighted sum of value elements for each query element
  results = SparseDenseMatMul()([attention, reshaped_value]); # results.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, value_dim // heads)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], x[1]))([results, shape]); # results.shape = (batch, heads, *other_dims, axial_dim_length, value_dim // heads)
  def get_inv_perm(origin_shape, axial_dim):
    perm = get_perm(origin_shape, axial_dim);
    return np.argsort(perm);
  results = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_inv_perm(origin_shape, axial_dim)})(results); # results.shape = (batch, heads, *origin_shape, value_dim // heads)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1, tf.shape(x)[-1])))(results); # results.shape = (batch, heads, query_length = np.prod(origin_shape), value_dim // heads)
  return tf.keras.Model(inputs = (query, key, value), outputs = results);

class BlockSparseAttention(tf.keras.Model):
  def __init__(self, num_heads, origin_shape = None, block = 32, local_blocks = 4, causal = True, sparse = 'local', look_backward_length = 5, **kwargs):
    super(BlockSparseAttention, self).__init__(**kwargs);
    self.origin_shape = origin_shape;
    self.block = block;
    self.local_blocks = local_blocks;
    self.causal = causal;
    self.num_heads = num_heads;
    self.block_transformer = BlocksparseTransformer(self.make_layout(), block_size = block, mask_callback = self.get_callback(sparse, look_backward_length), num_heads);
  def block_shape(self,):
    # block_shape = (l, h, w // block)
    cum_prod = 1;
    for i in range(len(self.origin_shape) - 1, 0, -1):
      cum_prod *= self.origin_shape[i];
      if cum_prod > self.block: break;
    assert cum_prod % self.block == 0;
    return (*self.origin_shape[:i], cum_prod // self.block);
  def idx_to_coord(self, idx):
    # convert idx in block_shape to coord
    shape = self.block_shape();
    shape_cum = tuple(np.flip(np.cumprod(np.flip(np.array(shape)))[:-1])) + (1,);
    coord = list();
    for i in range(len(shape)):
      coord.append(idx // shape[i]);
      idx %= shape_cum[i];
    return coord;
  def coord_to_idx(self, coord):
    # convert block_shape coord to idx
    shape = self.block_shape();
    shape_cum = tuple(np.flip(np.cumprod(np.flip(np.array(shape)))[:-1])) + (1,);
    idx = 0;
    for i in range(len(shape)):
      idx += coord[i] * shape_cum[i];
    return idx;
  def make_layout(self, ):
    assert np.prod(self.origin_shape) % self.block == 0;
    shape = self.block_shape();
    num_blocks = np.prod(shape);
    layout = np.zeros((num_blocks, num_blocks));
    # local layout
    for latter in range(num_blocks):
      for former in range(max(0, latter - self.local_blocks), (latter + 1 if self.causal == True else min(num_blocks, latter + self.local_blocks))):
        # 1) causal case: latter can only look formers backward in time as far as local_blocks number of blocks, can't look forward in time
        # 2) non causal case: latter can only look formers backward in time as far as local blocks number of blocks,
        #                     latter can only look forward in time as far as local blocks number of blocks
        layout[latter, former] = 1;
    # global layout
    for latter in range(num_blocks):
      latter_coord = self.idx_to_coord(latter);
      # calculate all visible formers
      for d in range(len(shape) - 1):
        for i in range(0, (latter_coord[d] + 1 if self.causal else shape[d])):
          # 1) causal case: latter can only look formers backward in time in every dimension of the block
          # 2) non causal case: latter can look forward and backward all elements in every dimension
          former_coord = latter_coord.copy();
          former_coord[d] = i;
          former = self.coord_to_idx(former_coord);
          layout[latter, former] = 1;
    return layout;
  def get_callback(self, sparse, look_backward_length = None):
    def get_mask(self, block_shape, head_idx, query_idx, key_idx, block_idx):
      if sparse in ['all', 'strided', 'fixed']:
        return np.tril(np.ones(block_shape)).astype(np.bool);
      elif sparse == 'local':
        return (np.tril(np.ones(block_shape)) - np.tril(np.ones(block_shape), -look_backward_length)).astype(np.bool);
      else:
        raise Exception('unknown sparse mode!');
    return get_mask
  def call(self, inputs):
    query = inputs[0]; # query.shape = (batch, query_length, key_dim)
    key = inputs[1]; # key.shape = (batch, key_length, key_dim)
    value = inputs[2]; # value.shape = (batch, key_length, value_dim)
    qk = self.block_transformer.query_key_op(query, key);
    attention = self.block_transformer.masked_softmax(qk, scale = 1. / tf.math.sqrt(tf.shape(query)[-1] / self.num_heads));
    results = self.block_transformer.weight_value_op(attention, value);
    
    # TODO

def MultiHeadAttention(key_dim, value_dim, num_heads, attn_type = 'full', sparse = None, look_backward_length = 5, origin_shape = (64, 64), axial_dim = -1):
  assert attn_type in ['full', 'axial', 'sparse'];
  query = tf.keras.Input((None, key_dim,)); # query.shape = (batch, query_length, key_dim)
  key = tf.keras.Input((None, key_dim,)); # key.shape = (batch, key_length, key_dim)
  value = tf.keras.Input((None, value_dim,)); # value.shape = (batch, key_length, value_dim)
  # 1) change to channels which can divided by num_heads
  query_dense = tf.keras.layers.Dense(units = key_dim // num_heads * num_heads)(query);
  key_dense = tf.keras.layers.Dense(units = key_dim // num_heads * num_heads)(key);
  value_dense = tf.keras.layers.Dense(units = value_dim // num_heads * num_heads)(value);
  # 2) split the dimension to form mulitiple heads
  query_splitted = tf.keras.layers.Reshape((-1, num_heads, key_dim // num_heads))(query_dense); # query_splitted.shape = (batch, query_length, num_heads, key_dim // num_heads)
  query_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(query_splitted); # query_splitted.shape = (batch, num_heads, query_length, key_dim // num_heads)
  key_splitted = tf.keras.layers.Reshape((-1, num_heads, key_dim // num_heads))(key_dense); # key_splitted.shape = (batch, key_length, num_heads, key_dim // num_heads)
  key_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(key_splitted); # key_splitted.shape = (batch, num_heads, key_length, key_dim // num_heads)
  value_splitted = tf.keras.layers.Reshape((-1, num_heads, value_dim // num_heads))(value_dense); # value_splitted.shape = (batch, key_length, num_heads, value_dim // num_heads)
  value_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(value_splitted); # value_splitted.shape = (batch, num_heads, key_length, value_dim // num_heads)
  if attn_type == 'full':
    attended = FullAttention(key_dim, value_dim, num_heads, sparse = sparse, look_backward_length = look_backward_length)([query_splitted, key_splitted, value_splitted]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  elif attn_type == 'axial':
    attended = AxialAttention(key_dim, value_dim, num_heads, origin_shape, axial_dim, sparse = sparse, look_backward_length = look_backward_length)([query_splitted, key_splitted, value_splitted]); # reults.shape = (batch, num_heads, query_length, value_dim // num_heads)
  elif attn_type == 'sparse':
    pass;
  else:
    raise Exception('invalid attention type!');
  attended = tf.keras.layers.Lambda(lambda x: tf,transpose(x, (0, 2, 1, 3)))(attended); # attended.shape = (batch, query_length, num_heads, value_dim // num_heads)
  concated = tf.keras.layers.Reshape((-1, value_dim))(attended); # concated.shape = (batch, query_length, value_dim)
  # 3) output
  results = tf.keras.layers.Dense(key_dim)(concated); # results.shape = (batch, query_length, key_dim)
  return tf.keras.Model(inputs = (query, key, value), outputs = results);

if __name__ == "__main__":

  query = np.random.normal(size = (4,3,150,10));
  key = np.random.normal(size = (4,3,50,10));
  value = np.random.normal(size = (4,3,50,100));
  mask = np.random.randint(low = 0, high = 2, size = (4,1,150,50));
  fullattention = FullAttention(30,300,3,sparse = None); # cross attention
  results = fullattention([query,key,value,mask]);
  print('cross attention', results.shape);
  query = np.random.normal(size = (4,3,50,10));
  fullattention = FullAttention(30,300,3,sparse = 'all'); # self attention
  results = fullattention([query,key,value]);
  print('self attention all', results.shape);
  fullattention = FullAttention(30,300,3,sparse = 'local'); # self attention
  results = fullattention([query,key,value]);
  print('self attention local', results.shape);
  fullattention = FullAttention(30,300,3,sparse = 'strided'); # self attention
  results = fullattention([query,key,value]);
  print('self attention strided', results.shape);
  query = np.random.normal(size = (4,3,150,10));
  key = np.random.normal(size = (4,3,150,10));
  value = np.random.normal(size = (4,3,150,100));
  axialattention = AxialAttention(30,300,3,origin_shape = (10,5,3), axial_dim = 1, sparse = 'all');
  results = axialattention([query,key,value]);
  print('self attention all', results.shape);
  axialattention = AxialAttention(30,300,3,origin_shape = (10,5,3), axial_dim = 1, sparse = 'local');
  results = axialattention([query,key,value]);
  print('self attention local', results.shape);
  axialattention = AxialAttention(30,300,3,origin_shape = (10,5,3), axial_dim = 1, sparse = 'strided');
  results = axialattention([query,key,value]);
  print('self attention strided', results.shape);
