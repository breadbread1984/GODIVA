#!/usr/bin/python3

import tensorflow as tf;

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
    self.enable_train;
    super(QuantizeEma, self).__init(**kwargs);
  def build(self, input_shape):
    self.cluster_mean = self.add_weight(shape = (self.embed_dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = True, name = 'cluster_mean');
    self.cluster_size = self.add_weight(shape = (self.n_embed,), dtype = tf.float32, initializer = tf.keras.initializers.Zeros(), trainable = True, name = 'cluster_size');
    self.cluster_sum = self.add_weight(shape = (self.embed_dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = True, name = 'cluster_sum');
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

def FullAttention(key_dim, value_dim, num_heads):
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query.shape = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length or 1, key_length)
  # 1) correlation matrix of query and key
  qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([query, key]); # qk.shape = (batch, heads, query_length, key_length)
  logits = tf.keras.layers.Lambda(lambda x, kd: x[0] / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)) + x[1] * -1e9, arguments = {'kd': key_dim // num_heads})([qk, mask]); # logits.shape = (batch, heads, query_length, key_length)
  attention = tf.keras.layers.Softmax()(logits); # attention.shape = (batch, heads, query_length, key_length)
  # 2) weighted sum of value elements for each query element
  attention = tf.keras.layers.Dropout()(attention);
  results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, value]); # results.shape = (batch, heads, query_length, value_dim // heads)
  return tf.keras.Model(inputs = (query, key, value, mask), outputs = results);

def MultiHeadAttention(key_dim, value_dim, num_heads, attn_type = 'full'):
  assert attn_type in ['full', 'axial', 'sparse'];
  query = tf.keras.Input((None, key_dim,)); # query.shape = (batch, query_length, key_dim)
  key = tf.keras.Input((None, key_dim,)); # key.shape = (batch, key_length, key_dim)
  value = tf.keras.Input((None, value_dim,)); # value.shape = (batch, key_length, value_dim)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length or 1, key_length)
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
    attended = FullAttention(key_dim, value_dim, num_heads)([query_splitted, key_splitted, value_splitted, mask]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  elif attn_type == 'axial':
    pass;
  elif attn_type == 'sparse':
    pass;
  else:
    raise Exception('invalid attention type!');
  attended = tf.keras.layers.Lambda(lambda x: tf,transpose(x, (0, 2, 1, 3)))(attended); # attended.shape = (batch, query_length, num_heads, value_dim // num_heads)
  concated = tf.keras.layers.Reshape((-1, value_dim))(attended); # concated.shape = (batch, query_length, value_dim)
  # 3) output
  results = tf.keras.layers.Dense(key_dim)(concated); # results.shape = (batch, query_length, key_dim)
  return tf.keras.Model(inputs = (query, key, value, mask), outputs = results);

if __name__ == "__main__":
  import numpy as np;
  tf.keras.backend.set_learning_phase(1);
  encoder = VQVAE_Encoder();
  decoder = VQVAE_Decoder();
  encoder.save_weights('encoder_weights.h5');
  decoder.save_weights('decoder_weights.h5');
  inputs = np.random.normal(size = (4,256,256,3));
  quantized_t, cluster_index_t, loss_t, quantized_b, cluster_index_b, loss_b = encoder(inputs);
  print(quantized_t.shape, cluster_index_t.shape, loss_t.shape);
  print(quantized_b.shape, cluster_index_b.shape, loss_b.shape);
  results = decoder([quantized_t, quantized_b]);
  print(results.shape);
  tf.keras.utils.plot_model(model = encoder, to_file = 'encoder.png', show_shapes = True, dpi = 64);
  tf.keras.utils.plot_model(model = decoder, to_file = 'decoder.png', show_shapes = True, dpi = 64);
  encoder = Encoder();
  decoder = Decoder(64,64);
  tf.keras.utils.plot_model(model = encoder, to_file = 'encoder.png', show_shapes = True, dpi = 64);
  tf.keras.utils.plot_model(model = decoder, to_file = 'decoder.png', show_shapes = True, dpi = 64);
