#!/usr/bin/python3

from os.path import join;
import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;

class Quantize(tf.keras.layers.Layer):
  def __init__(self, embed_dim = 128, n_embed = 10000, **kwargs):
    self.embed_dim = embed_dim;
    self.n_embed = n_embed;
    super(Quantize, self).__init__(**kwargs);
  def build(self, input_shape):
    # cluster_mean: cluster means which are used as codes for code book
    self.cluster_mean = self.add_weight(shape = (self.embed_dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = True, name = 'cluster_mean');
  def get_embed(self,):
    return self.cluster_mean;
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
  def get_embed(self,):
    return self.cluster_mean;
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
  def __init__(self, in_channels = 3, hidden_channels = 128, block_num = 2, res_channels = 32, embed_dim = 128, n_embed = 10000, quantize_type = 'original'):
    super(VQVAE_Trainer, self).__init__();
    self.encoder = VQVAE_Encoder(in_channels, hidden_channels, block_num, res_channels, embed_dim, n_embed, quantize_type);
    self.decoder = VQVAE_Decoder(in_channels, hidden_channels, block_num, res_channels, embed_dim);
  def call(self, inputs):
    quantized_t, cluster_index_t, loss_t, quantized_b, cluster_index_b, loss_b = self.encoder(inputs);
    recon = self.decoder([quantized_t, quantized_b]);
    loss = tf.keras.layers.Add()([loss_t, loss_b]);
    return recon, loss;

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

def FullAttention(key_dim, value_dim, num_heads, drop_rate = 0.2, causal = True):
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  if causal == True:
    y = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.shape(x)[2], dtype = tf.int32),(-1,1)),(1,tf.shape(x)[2])))(query);
    x = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.shape(x)[2], dtype = tf.int32),(1,-1)),(tf.shape(x)[2],1)))(query);
    mask = tf.keras.layers.Lambda(lambda x: tf.where(tf.math.greater_equal(x[0], x[1]), tf.ones_like(x[0]), tf.zeros_like(x[1])))([y,x]);
    mask = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(x[0], (1,1,tf.shape(x[0])[0],tf.shape(x[0])[1])), (tf.shape(x[1])[0],1,1,1)))([mask, query]);
  else:
    mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length, key_length)
  qk = MaskedDenseMatMul()([query, key, mask]);
  logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads, query_length, key_length)
  logits = Dense2Sparse()([logits, mask]); # logits.shape = (batch, heads, query_length, key_length)
  attention = tf.keras.layers.Lambda(lambda x: tf.sparse.softmax(x))(logits); # attention.shape = (batch, num_heads, query_length, key_length)
  # FIXME: uncomment the following line
  #attention = SparseDropout(drop_rate = drop_rate)(attention); # attention.shape = (batch, num_heads, query_length, key_length)
  # 2) weighted sum of value elements for each query element
  results = SparseDenseMatMul()([attention, value]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  return tf.keras.Model(inputs = (query, key, value) if causal == True else (query, key, value, mask), outputs = results);

def AxialAttention(key_dim, value_dim, num_heads, drop_rate = 0.5, origin_shape = None, axial_dim = 0):
  # NOTE: this attention can only apply to self attention, but cross attention.
  # in other words, query_length = key_length must hold
  # NOTE: leave one dim as seq_length, merge the other dims with heads.
  # for example key.shape = (batch, heads, h, w, c, dim) and axial_dim = -2
  # key.shape becomes (batch, new_heads = heads * h * c, seq_length = w, dim),
  # the self attention matrix become w x w, rather than (h * w * c) x (h * w * c)
  assert type(origin_shape) in [list, tuple];
  assert 0 <= axial_dim < 3 or -3 <= axial_dim < 0;
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query.shape = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  reshaped_query = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], key_dim // num_heads))(query);
  reshaped_key = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], key_dim // num_heads))(key);
  reshaped_value = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], value_dim // num_heads))(value);
  def get_perm(axial_dim):
    dims = np.arange(2 + 3 + 1); # batch x heads x *origin_shape x dim
    chosed_dim = 2 + axial_dim if axial_dim >= 0 else 2 + 3 + axial_dim;
    index = dims.tolist().index(chosed_dim);
    dims[index], dims[-2] = dims[-2], dims[index];
    return dims;
  reshaped_query = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_query);
  reshaped_query = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_query); # query.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  reshaped_key = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_key);
  reshaped_key = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_key); # key.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  reshaped_value = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_value);
  shape = tf.keras.layers.Lambda(lambda x: tf.shape(x))(reshaped_value);
  reshaped_value = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_value); # value.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  # 1) correlation matrix of query and key
  qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([reshaped_query, reshaped_key]);
  logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  attention = tf.keras.layers.Softmax()(logits); # attention.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  attention = tf.keras.layers.Dropout(rate = drop_rate)(attention);
  # 2) weighted sum of value elements for each query element
  results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, reshaped_value]); # results.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, value_dim // heads)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], x[1]))([results, shape]); # results.shape = (batch, heads, *other_dims, axial_dim_length, value_dim // heads)
  def get_inv_perm(axial_dim):
    perm = get_perm(axial_dim);
    return np.argsort(perm);
  results = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_inv_perm(axial_dim)})(results); # results.shape = (batch, heads, *origin_shape, value_dim // heads)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1, tf.shape(x)[-1])))(results); # results.shape = (batch, heads, query_length = np.prod(origin_shape), value_dim // heads)
  return tf.keras.Model(inputs = (query, key, value), outputs = results);

def SparseAttention(key_dim, value_dim, num_heads, drop_rate = 0.5, origin_shape = None, causal = True, local_blocks = 3):
  assert type(origin_shape) in [list, tuple];
  # NOTE: local is the first mask for the strided head
  # NOTE: strided is the second mask for the strided head
  def idx_to_coord(idx):
    # convert idx in block_shape to coord
    shape = origin_shape;
    shape_cum = tuple(np.flip(np.cumprod(np.flip(np.array(shape)))[:-1])) + (1,);
    coord = list();
    for i in range(len(shape)):
      coord.append(idx // shape_cum[i]);
      idx %= shape_cum[i];
    return coord;
  def coord_to_idx(coord):
    # convert block_shape coord to idx
    shape = origin_shape;
    shape_cum = tuple(np.flip(np.cumprod(np.flip(np.array(shape)))[:-1])) + (1,);
    idx = 0;
    for i in range(len(shape)):
      idx += coord[i] * shape_cum[i];
    return idx;
  def make_layout():
    shape = origin_shape;
    num_blocks = np.prod(shape);
    layout = np.zeros((num_blocks, num_blocks), dtype = np.bool);
    # 1) first mask for the strided head
    for latter in range(num_blocks):
      for former in range(max(0, latter - local_blocks), (latter + 1 if causal == True else min(num_blocks, latter + local_blocks))):
        # 1) causal case: latter can only look formers backward in time as far as local_blocks number of blocks, can't look forward in time
        # 2) non causal case: latter can only look formers backward in time as far as local blocks number of blocks,
        #                     latter can only look forward in time as far as local blocks number of blocks
        layout[latter, former] = True;
    # 2) second mask for the strided head
    for latter in range(num_blocks):
      latter_coord = idx_to_coord(latter);
      # calculate all visible formers
      for d in range(len(shape) - 1):
        for i in range(0, (latter_coord[d]+1 if causal else shape[d])):
          # 1) causal case: latter can only look formers backward in time in every dimension of the block
          # 2) non causal case: latter can look forward and backward all elements in every dimension
          former_coord = latter_coord.copy();
          former_coord[d] = i;
          former = coord_to_idx(former_coord);
          if former >= num_blocks:
            print(former_coord);
          layout[latter, former] = True;
    layout = np.reshape(layout, (1,1,num_blocks,num_blocks)); # layout.shape = (1, 1, query_length, key_length)
    return layout;
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query.shape = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  mask = tf.keras.layers.Lambda(lambda x, m: tf.tile(m,(tf.shape(x)[0],1,1,1)), arguments = {'m': make_layout()})(query); # mask.shape = (batch, 1, query_length, key_length)
  qk = MaskedDenseMatMul()([query, key, mask]);
  logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads, query_length, key_length)
  logits = Dense2Sparse()([logits, mask]); # logits.shape = (batch, heads, query_length, key_length)
  attention = tf.keras.layers.Lambda(lambda x: tf.sparse.softmax(x))(logits); # attention.shape = (batch, num_heads, query_length, key_length)
  # FIXME: uncomment the following line
  #attention = SparseDropout(drop_rate = drop_rate)(attention); # attention.shape = (batch, num_heads, query_length, key_length)
  # 2) weighted sum of value elements for each query element
  results = SparseDenseMatMul()([attention, value]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  return tf.keras.Model(inputs = (query, key, value), outputs = results);

def MultiHeadAttention(key_dim, value_dim, num_heads, attn_type = 'full', **kwargs):
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
    if kwargs['causal'] == False:
      mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length, key_length)
    attended = FullAttention(key_dim, value_dim, num_heads, kwargs['drop_rate'], kwargs['causal'])([query_splitted, key_splitted, value_splitted] if kwargs['causal'] == True else [query_splitted, key_splitted, value_splitted, mask]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  elif attn_type == 'axial':
    attended = AxialAttention(key_dim, value_dim, num_heads, kwargs['drop_rate'], kwargs['origin_shape'], kwargs['axial_dim'])([query_splitted, key_splitted, value_splitted]); # reults.shape = (batch, num_heads, query_length, value_dim // num_heads)
  elif attn_type == 'sparse':
    attended = SparseAttention(key_dim, value_dim, num_heads, kwargs['drop_rate'], kwargs['origin_shape'], kwargs['causal'], kwargs['local_blocks'])([query_splitted, key_splitted, value_splitted]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  else:
    raise Exception('invalid attention type!');
  attended = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(attended); # attended.shape = (batch, query_length, num_heads, value_dim // num_heads)
  concated = tf.keras.layers.Reshape((-1, value_dim))(attended); # concated.shape = (batch, query_length, value_dim)
  # 3) output
  results = tf.keras.layers.Dense(key_dim)(concated); # results.shape = (batch, query_length, key_dim)
  return tf.keras.Model(inputs = (query, key, value) if attn_type != 'full' or kwargs['causal'] == True else (query, key, value, mask), outputs = results);

def PositionalEncoding(d_model):
  # 1) inputs
  inputs = tf.keras.Input((None, d_model)); # inputs.shape = (batch, length, dimension)
  # 2) position info
  positions = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.range(tf.cast(tf.shape(x)[1], dtype = tf.float32), dtype = tf.float32),1))(inputs); # positions.shape = (length, 1)
  # 3) dimension info
  j = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.range(tf.cast(tf.shape(x)[2], dtype = tf.float32), dtype = tf.float32),0))(inputs); # j.shape = (1, dimension)
  i = tf.keras.layers.Lambda(lambda x: x // 2)(j);                                                                                           # i.shape = (1, dimension)
  power = tf.keras.layers.Lambda(lambda x: 2 * x[0] / tf.cast(tf.shape(x[1])[2], dtype = tf.float32))([i, inputs]);                          # power.shape = (1, dimension)
  # 4) position & dimension info
  angles = tf.keras.layers.Lambda(lambda x: x[0] / tf.math.pow(10000.,x[1]))([positions, power]);                                            # angles.shape = (length, dimension)
  sines = tf.keras.layers.Lambda(lambda x: tf.math.sin(x[:,0::2]))(angles);                                                                  # sines.shape = (length, dimension // 2)
  cosines = tf.keras.layers.Lambda(lambda x: tf.math.cos(x[:,1::2]))(angles);                                                                # cosines.shape = (length, dimension // 2)
  pos_encoding = tf.keras.layers.Concatenate()([sines, cosines]);                                                                            # pos_encoding.shape = (length, dimension)
  pos_encoding = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x[0],0), (tf.shape(x[1])[0], 1, 1)))([pos_encoding, inputs]);       # pos_encoding.shape = (batch, length, dimension)
  # 5) positional & embedding
  results = tf.keras.layers.Add()([inputs, pos_encoding]);                                                                                   # results.shape = (batch, length, dimension)
  return tf.keras.Model(inputs = inputs, outputs = results);

def EncoderLayer(hidden_dim = 1024, num_heads = 16, **kwargs):
  # NOTE: this sparse transformer residual block is for decoder only transformer
  inputs = tf.keras.Input((None, hidden_dim,)); # inputs.shape = (batch, hidden_length, hidden_dim)
  short = inputs;
  results = tf.keras.layers.LayerNormalization()(inputs); # results.shape = (batch, hidden_length, hidden_dim)
  results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, attn_type = 'full', **kwargs)([results, results, results]); # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results); # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Add()([results, short]); # results.shape = (batch, hidden_length, hidden_dim)
  short = results;
  results = tf.keras.layers.LayerNormalization()(results); # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Dense(hidden_dim * 4)(results); # results.shape = (batch, hidden_length, 4 * hidden_dim)
  results = tfa.layers.GELU()(results); # results.shape = (batch, hidden, 4 * hidden_dim)
  results = tf.keras.layers.Dense(hidden_dim)(results);  # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results); # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Add()([results, short]); # results.shape = (batch, hidden_length, hidden_dim)
  return tf.keras.Model(inputs = inputs, outputs = results);

def TransEncoder(num_layers = 2, hidden_dim = 1024, num_heads = 16, **kwargs):
  inputs = tf.keras.Input((None, hidden_dim));
  embeddings = PositionalEncoding(hidden_dim)(inputs);
  outputs = tf.keras.layers.Dropout(rate = kwargs['drop_rate'])(embeddings);
  for i in range(num_layers):
    outputs = EncoderLayer(hidden_dim, num_heads, drop_rate = kwargs['drop_rate'], causal = True)(outputs);
  return tf.keras.Model(inputs = inputs, outputs = outputs);

def DecoderLayer(hidden_dim = 1024, num_heads = 16, **kwargs):
  inputs = tf.keras.Input((None, hidden_dim,));
  code = tf.keras.Input((None, hidden_dim));
  short = inputs;
  results = tf.keras.layers.LayerNormalization()(inputs);
  for i in range(4):
    results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, attn_type = 'axial', drop_rate = kwargs['drop_rate'], origin_shape = kwargs['origin_shape'], axial_dim = -3)([results, results, results]);
    results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, attn_type = 'axial', drop_rate = kwargs['drop_rate'], origin_shape = kwargs['origin_shape'], axial_dim = -2)([results, results, results]);
    results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, attn_type = 'axial', drop_rate = kwargs['drop_rate'], origin_shape = kwargs['origin_shape'], axial_dim = -1)([results, results, results]);
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results);
  results = tf.keras.layers.Add()([results, short]);
  mask = tf.keras.layers.Lambda(lambda x: tf.ones((tf.shape(x[0])[0],1,tf.shape(x[0])[1], tf.shape(x[1])[1])))([inputs, code]);
  short = results;
  results = tf.keras.layers.LayerNormalization()(results);
  results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, 'full', drop_rate = kwargs['drop_rate'], causal = False)([results, code, code, mask]);
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results);
  results = tf.keras.layers.Add()([results, short]);
  short = results;
  results = tf.keras.layers.LayerNormalization()(results);
  results = tf.keras.layers.Dense(hidden_dim * 4)(results);
  results = tfa.layers.GELU()(results);
  results = tf.keras.layers.Dense(hidden_dim)(results);
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results);
  results = tf.keras.layers.Add()([results, short]);
  return tf.keras.Model(inputs = (inputs, code), outputs = results);

def TransDecoder(num_layers = 2, hidden_dim = 1024, num_heads = 16, **kwargs):
  inputs = tf.keras.Input((None, hidden_dim));
  code = tf.keras.Input((None, hidden_dim));
  embeddings = PositionalEncoding(hidden_dim)(inputs);
  outputs = tf.keras.layers.Dropout(rate = kwargs['drop_rate'])(embeddings);
  for i in range(num_layers):
    outputs = DecoderLayer(hidden_dim, num_heads, drop_rate = kwargs['drop_rate'], origin_shape = kwargs['origin_shape'])([outputs, code]);
  return tf.keras.Model(inputs = (inputs, code), outputs = outputs);

def Transformer(encoder_layers = 2, decoder_layers = 2, hidden_dim = 128, num_heads = 16, origin_shape = (64, 64), text_vocab_size = None, video_vocab_size = 10000, **kwargs):
  text_inputs = tf.keras.Input((None,)); # inputs.shape = (batch, text_length)
  # INFO: to avoid repeat calculating embedding of leading frames, the input uses code from VQVAE, but leading frames
  # NOTE: video_top_inputs.shape[1] = origin_shape[1] // 8 * origin_shape[2] // 8 * frame_number
  # NOTE: video_bottom_inputs.shape[1] = origin_shape[1] // 4 * origin_shape[2] // 4 * frame_number
  video_inputs = tf.keras.Input((None,)); # video_top_inputs.shape = (batch, frame * 64 * 64 + 2,)
  
  text_embed = tf.keras.layers.Embedding(text_vocab_size, hidden_dim)(text_inputs);
  text_embed = tf.keras.layers.Lambda(lambda x, d: tf.math.sqrt(tf.cast(d, dtype = tf.float32)) * x, arguments = {'d': hidden_dim})(text_embed);
  video_embed = tf.keras.layers.Embedding(video_vocab_size, hidden_dim)(video_inputs);
  video_embed = tf.keras.layers.Lambda(lambda x, d: tf.math.sqrt(tf.cast(d, dtype = tf.float32)) * x, arguments = {'d': hidden_dim})(video_embed);
  
  text_code = TransEncoder(encoder_layers, hidden_dim, num_heads, drop_rate = kwargs['drop_rate'])(text_embed); # text_code.shape = (batch, text_length, hidden_dim)
  video_code = TransDecoder(decoder_layers, hidden_dim, num_heads, drop_rate = kwargs['drop_rate'], origin_shape = (origin_shape[0], origin_shape[1]))([video_embed, text_code]);
  video_pred = tf.keras.layers.Dense(units = video_vocab_size, activation = tf.keras.activations.softmax)(video_code);
  return tf.keras.Model(inputs = (text_inputs, video_inputs), outputs = video_pred);

class GODIVA(tf.keras.Model):
  def __init__(self, vq_type = 'ema_update', vq_encoder_model = join('models', 'encoder_d128_c10000_64x64.h5'), vq_decoder_model = join('models', 'decoder_d128_c10000_64x64.h5'), origin_shape = (64, 64), video_length = 16, text_vocab_size = None, video_vocab_size = 10000, **kwargs):
    super(GODIVA, self).__init__(**kwargs);
    self.origin_shape = origin_shape;
    self.video_length = video_length;
    self.video_vocab_size = video_vocab_size;
    # NOTE: tokens for vqvae are in range [0, video_vocab_size - 1]
    # NOTE: the following two tokens are for start of string (SOS) and end of string (EOS)
    self.SOS = self.video_vocab_size;
    self.EOS = self.video_vocab_size + 1;
    self.top_frame_token_num = self.origin_shape[0] // 8 * self.origin_shape[1] // 8;
    self.bottom_frame_token_num = self.origin_shape[0] // 4 * self.origin_shape[1] // 4;
    self.encoder = tf.keras.models.load_model(vq_encoder_model, custom_objects = {'Quantize': Quantize, 'QuantizeEma': QuantizeEma});
    self.decoder = tf.keras.models.load_model(vq_decoder_model);
    self.encoder.trainable = False;
    self.decoder.trainable = False;
    self.top_transformer = Transformer(origin_shape = (origin_shape[0] // 8, origin_shape[1] // 8), text_vocab_size = text_vocab_size, video_vocab_size = video_vocab_size + 2, drop_rate = 0.2);
    self.bottom_transformer = Transformer(origin_shape = (origin_shape[0] // 4, origin_shape[1] // 4), text_vocab_size = text_vocab_size, video_vocab_size = video_vocab_size + 2, drop_rate = 0.2);
  def call(self, inputs):
    # NOTE: top_tokens of the first frame is full of SOS tokens
    # NOTE: bottom_tokens of the first frame is full of SOS tokens
    top_tokens = tf.ones((tf.shape(inputs)[0], self.top_frame_token_num), dtype = tf.int64) * self.SOS; # top_tokens.shape = (batch, (origin_shape // 8) ** 2)
    bottom_tokens = tf.ones((tf.shape(inputs)[0], self.bottom_frame_token_num), dtype = tf.int64) * self.SOS; # bottom_tokens.shape = (batch, (origin_shape // 4) ** 2)
    for i in range(self.video_length + 1):
      top_pred = self.top_transformer([inputs, top_tokens]); # top_pred.shape = (batch, length, video_vocab_size)
      tokens = tf.math.argmax(top_pred, axis = -1); # tokens.shape = (batch, length)
      top_tokens = tf.concat([top_tokens, tokens[:,-self.top_frame_token_num:]], axis = 1); # top_tokens.shape = (batch, length)
    for i in range(self.video_length + 1):
      bottom_pred = self.bottom_transformer([inputs, bottom_tokens]);
      tokens = tf.math.argmax(bottom_pred, axis = -1); # tokens.shape = (batch, length)
      bottom_tokens = tf.concat([bottom_tokens, tokens[:,-self.bottom_frame_token_num:]], axis = 1); # bottom_tokens.shape = (batch, 1+length)
    return top_tokens, bottom_tokens;
  def decode(self, top_tokens, bottom_tokens):
    top_embed_mat = self.encoder.layers[4].get_embed();
    bottom_embed_mat = self.encoder.layers[8].get_embed();
    top_tokens = top_tokens[:,self.top_frame_token_num:-self.top_frame_token_num]; # strip start and end tokens
    bottom_tokens = bottom_tokens[:,self.bottom_frame_token_num:-self.bottom_frame_token_num]; # strip start and end tokens
    top_embeddings = tf.nn.embedding_lookup(tf.transpose(top_embed_mat), top_tokens); # top_embeddings.shape = (batch, length * size//8 * size//8, embed_dim)
    bottom_embeddings = tf.nn.embedding_lookup(tf.transpose(bottom_embed_mat), bottom_tokens); # bottom_embedding.shape = (batch, length * size//4 * size//4, embed_dim)
    top_embeddings = tf.reshape(top_embeddings, (tf.shape(top_embeddings)[0], self.video_length, self.origin_shape[0]//8, self.origin_shape[1]//8, tf.shape(top_embeddings)[-1])); # top_embeddings.shape = (batch, length, size//8, size//8, embed_dim)
    bottom_embeddings = tf.reshape(bottom_embeddings, (tf.shape(bottom_embeddings)[0], self.video_length, self.origin_shape[0]//4, self.origin_shape[1]//4, tf.shape(bottom_embeddings)[-1])); # bottom_embeddings.shape = (batch, length, size//4, size//4, embed_dim)
    video = list();
    for i in range(self.video_length):
      frame = self.decoder([top_embeddings[:, i, ...], bottom_embeddings[:, i, ...]]); # frame.shape = (batch, 64, 64, 3)
      video.append(frame);
    video = tf.stack(video, axis = 1); # video.shape = (batch, seq_length, 64, 64, 3)
    return video;

if __name__ == "__main__":

  tokens = np.random.randint(low = 0, high = 10, size = (1, 34));
  godiva = GODIVA(text_vocab_size = 10);
  top_tokens, bottom_tokens = godiva(tokens);
  print(top_tokens.shape, bottom_tokens.shape);
  godiva.save_weights('godiva.h5');
