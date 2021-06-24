#!/usr/bin/python3

import tensorflow as tf;

class Quantize(tf.keras.layers.Layer):
  def __init__(self, embed_dim = 128, n_embed = 10000, decay = 0.99, eps = 1e-5, enable_train = False, **kwargs):
    self.embed_dim = embed_dim;
    self.n_embed = n_embed;
    self.decay = decay;
    self.eps = eps;
    self.enable_train = enable_train;
    super(Quantize, self).__init__(**kwargs);
  def build(self, input_shape):
    # NOTE: all this weights are not trainable, because they are managed manually
    # cluster_mean: cluster means which are used as codes for code book
    # cluster_size: how many samples falling in each cluster
    # cluster_sum: the respective sum of samples falling in each cluster
    self.cluster_mean = self.add_weight(shape = (self.embed_dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = False, name = 'cluster_mean');
    self.cluster_size = self.add_weight(shape = (self.n_embed,), dtype = tf.float32, initializer = tf.keras.initializers.Zeros(), trainable = False, name = 'cluster_size');
    self.cluster_sum = self.add_weight(shape = (self.embed_dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = False, name = 'cluster_sum');
    self.cluster_sum.assign(self.cluster_mean);
  def call(self, inputs):
    samples = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, self.embed_dim,)))(inputs); # samples.shape = (n_sample, dim)
    # dist = (X - cluster_mean)^2 = X' * X - 2 * X' * Embed + trace(Embed' * Embed),  dist.shape = (n_sample, n_embed), euler distances to cluster_meanding vectors
    dist = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.pow(x[0],2), axis = 1, keepdims = True) - 2 * tf.linalg.matmul(x[0], x[1]) + tf.math.reduce_sum(tf.math.pow(x[1],2), axis = 0, keepdims = True))([samples, self.cluster_mean]);
    cluster_index = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = 1))(dist); # cluster_index.shape = (n_sample)
    quantize = tf.keras.layers.Lambda(lambda x: tf.nn.embedding_lookup(tf.transpose(x[0]), x[1]))([self.cluster_mean, cluster_index]); # quantize.shape = (n_sample, dim)
    if tf.keras.backend.learning_phase() == 1 and self.enable_train:
      # NOTE: code book is updated during forward propagation
      cluster_index_onehot = tf.keras.layers.Lambda(lambda x, n: tf.one_hot(x, n), arguments = {'n': self.n_embed})(cluster_index); # cluster_index_onehot.shape = (n_sample, n_embed)
      cluster_size = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = 0))(cluster_index_onehot); # cluster_size.shape = (n_embed)
      cluster_sum = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_a = True))([samples, cluster_index_onehot]); # cluster_sum.shape = (dim, n_embed)
      updated_cluster_size = tf.keras.layers.Lambda(lambda x, d: x[0] * d + x[1] * (1 - d), arguments = {'d': self.decay})([self.cluster_size, cluster_size]); # updated_cluster_size.shape = (n_embed)
      updated_cluster_sum = tf.keras.layers.Lambda(lambda x, d: x[0] * d + x[1] * (1 - d), arguments = {'d': self.decay})([self.cluster_sum, cluster_sum]); # updated_cluster_sum.shape = (dim, n_embed)
      self.cluster_size.assign(updated_cluster_size);
      self.cluster_sum.assign(updated_cluster_sum);
      n_sample = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x))(self.cluster_size); # n_sample.shape = ()
      cluster_size = tf.keras.layers.Lambda(lambda x, e, n: (x[0] + e) * x[1] / (x[1] + n * e), arguments = {'e': self.eps, 'n': self.n_embed})([self.cluster_size, n_sample]); # cluster_size.shape = (n_embed)
      cluster_mean = tf.keras.layers.Lambda(lambda x: x[0] / x[1])([self.cluster_sum, self.cluster_size]); # cluster_mean.shape = (dim, n_embed)
      self.cluster_mean.assign(cluster_mean);
    quantize = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (tf.shape(x[1])[0], tf.shape(x[1])[1], tf.shape(x[1])[2], tf.shape(x[0])[-1])))([quantize, inputs]); # quantize.shape = (batch, h, w, dim)
    cluster_index = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (tf.shape(x[1])[0], tf.shape(x[1])[1], tf.shape(x[1])[2],)))([cluster_index, inputs]); # cluster_index.shape = (batch, h, w)
    diff = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.math.pow(x[0] - x[1], 2)))([inputs, quantize]); # diff.shape = (n_sample,)
    return quantize, cluster_index, diff;
  def set_trainable(self, trainable = True):
    self.train_quantize = trainable;
  def get_config(self):
    config = super(Quantize, self).get_config();
    config['embed_dim'] = self.embed_dim;
    config['n_embed'] = self.n_embed;
    config['decay'] = self.decay;
    config['eps'] = self.eps;
    config['enable_train'] = self.enable_train;
    return config;
  @classmethod
  def from_config(cls, config):
    self.embed_dim = config['embed_dim'];
    self.n_embed = config['n_embed'];
    self.decay = config['decay'];
    self.eps = config['eps'];
    self.enable_train = config['enable_train'];
    return cls(**config);

def Encoder(in_channels = 3, out_channels = 128, block_num = 2, res_channels = 32, stride = 4, name = 'encoder'):
  assert stride in [2, 4];
  inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch, height, width, in_channels)
  if stride == 4:
    results = tf.keras.layers.Conv2D(out_channels // 2, (4,4), strides = (2,2), padding = 'same')(inputs);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(out_channels, (4,4), strides = (2,2), padding = 'same')(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same')(results);
  elif stride == 2:
    results = tf.keras.layers.Conv2D(out_channels // 2, (4,4), strides = (2,2), padding = 'same')(inputs);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same')(results);
  else:
    raise Exception('invalid stride option');
  for i in range(block_num):
    short = results;
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(res_channels, (3,3), padding = 'same')(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(out_channels, (1,1), padding = 'same')(results);
    results = tf.keras.layers.Add()([results, short]);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = inputs, outputs = results, name = name);

def Decoder(in_channels, out_channels, hidden_channels, block_num, res_channels = 32, strides = 4, name = 'decoder'):
  assert strides in [2, 4];
  inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch, height, width, in_channels)
  results = tf.keras.layers.Conv2D(hidden_channels, (3,3), padding = 'same')(inputs);
  for i in range(block_num):
    short = results;
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(res_channels, (3,3), padding = 'same')(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(hidden_channels, (1,1), padding = 'same')(results);
    results = tf.keras.layers.Add()([results, short]);
  results = tf.keras.layers.ReLU()(results);
  if strides == 4:
    results = tf.keras.layers.Conv2DTranspose(hidden_channels // 2, (4,4), strides = (2,2), padding = 'same')(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(out_channels, (4,4), strides = (2,2), padding = 'same')(results);
  elif strides == 2:
    results = tf.keras.layers.Conv2DTranspose(out_channels, (4,4), strides = (2,2), padding = 'same')(results);
  else:
    raise Exception('invalid stride option');
  return tf.keras.Model(inputs = inputs, outputs = results, name = name);

def VQVAE_Encoder(in_channels = 3, hidden_channels = 128, block_num = 2, res_channels = 32, embed_dim = 64, n_embed = 512, train_quantize = False, name = 'encoder'):
  inputs = tf.keras.Input((None, None, in_channels));
  enc_b = Encoder(in_channels, hidden_channels, block_num, res_channels, 4, name = 'bottom_encoder')(inputs); # enc_b.shape = (batch, h/4, w/4, hidden_channels)
  enc_t = Encoder(hidden_channels, hidden_channels, block_num, res_channels, 2, name = 'top_encoder')(enc_b); # enc_t.shape = (batch, h/8, w/8, hidden_channels)
  results = tf.keras.layers.Conv2D(embed_dim, (1,1))(enc_t); # results.shape = (batch, h/8, w/8, embed_dim)
  quantized_t, cluster_index_t, diff_t = Quantize(embed_dim, n_embed, enable_train = train_quantize, name = 'top_quantize')(results); # quantized_t.shape = (batch, h/8, w/8, embed_dim)
  dec_t = Decoder(embed_dim, embed_dim, hidden_channels, block_num, res_channels, 2)(quantized_t); # dec_t.shape = (batch, h/4, w/4, embed_dim)
  enc_b = tf.keras.layers.Concatenate(axis = -1)([dec_t, enc_b]); # enc_b.shape = (bath, h/4, w/4, embed_dim + hidden_channels)
  results = tf.keras.layers.Conv2D(embed_dim, (1,1))(enc_b); # results.shape = (batch, h/4, w/4, embed_dim)
  quantized_b, cluster_index_b, diff_b = Quantize(embed_dim, n_embed, enable_train = train_quantize, name = 'bottom_quantize')(results); # quantized_b.shape = (batch, h/4, w/4, embed_dim)
  return tf.keras.Model(inputs = inputs, outputs = (quantized_t, cluster_index_t, diff_t, quantized_b, cluster_index_b, diff_b), name = name);

def VQVAE_Decoder(in_channels = 3, hidden_channels = 128, block_num = 2, res_channels = 32, embed_dim = 64, name = 'decoder'):
  quantized_t = tf.keras.Input((None, None, embed_dim)); # quantized_t.shape = (batch, h/8, w/8, embed_dim)
  quantized_b = tf.keras.Input((None, None, embed_dim)); # quantized_b.shape = (batch, h/4, w/4, embed_dim)
  results = tf.keras.layers.Conv2DTranspose(embed_dim, (4,4), strides = (2,2), padding = 'same')(quantized_t); # results.shape = (batch, h/4, w/4, embed_dim)
  results = tf.keras.layers.Concatenate(axis = -1)([results, quantized_b]); # results.shape = (batch, h/4, w/4, 2 * embed_dim)
  results = Decoder(2 * embed_dim, in_channels, hidden_channels, block_num, res_channels, 4)(results); # results.shape = (batch, h, w, 3)
  return tf.keras.Model(inputs = (quantized_t, quantized_b), outputs = results, name = name);

class VQVAE_Trainer(tf.keras.Model):
  def __init__(self, in_channels = 3, hidden_channels = 128, block_num = 2, res_channels = 32, embed_dim = 64, n_embed = 512):
    super(VQVAE_Trainer, self).__init__();
    self.encoder = VQVAE_Encoder(in_channels, hidden_channels, block_num, res_channels, embed_dim, n_embed, True);
    self.decoder = VQVAE_Decoder(in_channels, hidden_channels, block_num, res_channels, embed_dim);
  def call(self, inputs):
    quantized_t, cluster_index_t, diff_t, quantized_b, cluster_index_b, diff_b = self.encoder(inputs);
    recon = self.decoder([quantized_t, quantized_b]);
    diff = tf.keras.layers.Add(name = 'diff')([diff_t, diff_b]);
    return recon, diff;

if __name__ == "__main__":
  import numpy as np;
  tf.keras.backend.set_learning_phase(1);
  encoder = VQVAE_Encoder(train_quantize = False);
  decoder = VQVAE_Decoder();
  encoder.save('encoder.h5');
  decoder.save('decoder.h5');
  inputs = np.random.normal(size = (4,256,256,3));
  quantized_t, cluster_index_t, diff_t, quantized_b, cluster_index_b, diff_b = encoder(inputs);
  print(quantized_t.shape, cluster_index_t.shape, diff_t.shape);
  print(quantized_b.shape, cluster_index_b.shape, diff_b.shape);
  results = decoder([quantized_t, quantized_b]);
  print(results.shape);
