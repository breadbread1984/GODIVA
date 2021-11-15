#!/usr/bin/python3

from math import log2;
import tensorflow as tf;

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
    return outputs, loss;
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
    return outputs, 0.25 * e_loss;
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

def Encoder(in_channels = 3, out_channels = 128, downsample_rate = 4, res_layers = 4, res_channels = 32, **kwargs):
  assert round(log2(downsample_rate),0) == log2(downsample_rate);
  inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch, height, width, in_channels)
  results = inputs;
  # 1) downsample
  for i in range(int(log2(downsample_rate))):
    results = tf.keras.layers.Conv2D(out_channels, (4,4), strides = (2,2), padding = 'same', activation = tf.keras.activations.relu)(results);
  results = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same')(results);
  # 2) resnet blocks
  for i in range(res_layers):
    short = results;
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(res_channels, (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
    results = tf.keras.layers.Conv2D(out_channels, (1,1), padding = 'same')(results);
    results = tf.keras.layers.Add()([results, short]);
  return tf.keras.Model(inputs = inputs, outputs = results, **kwargs);

def Decoder(in_channels, out_channels, upsample_rate = 4, res_layers = 4, res_channels = 32, **kwargs):
  assert round(log2(upsample_rate),0) == log2(upsample_rate);
  inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch, height, width, in_channels)
  results = tf.keras.layers.Conv2D(out_channels, (1,1))(inputs); # results.shape = (batch, height, width, out_channels)
  # 1) resnet blocks
  for i in range(res_layers):
    short = results;
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(res_channels, (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
    results = tf.keras.layers.Conv2D(out_channels, (1,1), padding = 'same')(results);
    results = tf.keras.layers.Add()([results, short]);
  # 2) upsample
  for i in range(int(log2(upsample_rate))):
    results = tf.keras.layers.Conv2DTranspose(out_channels, (4,4), strides = (2,2), padding = 'same', activation = tf.keras.activations.relu if i != int(log2(upsample_rate)) - 1 else tf.keras.activations.sigmoid)(results);
  results = tf.keras.layers.Lambda(lambda x: x - 0.5)(results); # range in [-0.5, 0.5]
  return tf.keras.Model(inputs = inputs, outputs = results, **kwargs);

def VQVAE_Encoder(in_channels = 3, hidden_channels = 128, res_layer = 4, res_channels = 32, embed_dim = 64, n_embed = 512, quantize_type = 'original', **kwargs):
  quantize_type in ['original', 'ema_update'];
  inputs = tf.keras.Input((None, None, in_channels));
  results = Encoder(in_channels, hidden_channels, 4, res_layer, res_channels, name = 'bottom_encoder')(inputs); # results.shape = (batch, h/4, w/4, hidden_channels)
  results = tf.keras.layers.Conv2D(embed_dim, (1,1))(results); # results.shape = (batch, h/4, w/4, embed_dim)
  if quantize_type == 'original':
    quantized, loss = Quantize(embed_dim, n_embed, name = 'top_quantize')(results); # quantized_t.shape = (batch, h/8, w/8, embed_dim)
  else:
    quantized, loss = QuantizeEma(embed_dim, n_embed, name = 'top_quantize')(results); # quantized_t.shape = (batch, h/8, w/8, embed_dim)
  return tf.keras.Model(inputs = inputs, outputs = (quantized, loss), **kwargs);

def VQVAE_Decoder(in_channels = 3, hidden_channels = 128, res_layer = 4, res_channels = 32, embed_dim = 64, **kwargs):
  quantized = tf.keras.Input((None, None, embed_dim)); # quantized_t.shape = (batch, h/4, w/4, embed_dim)
  results = tf.keras.layers.Conv2D(hidden_channels, (1,1))(quantized); # results.shape = (batch, h/4, w/4, hidden_channels)
  results = Decoder(hidden_channels, in_channels, 4, res_layer, res_channels, name = 'decoder')(results); # results.shape = (batch, h, w, 3)
  return tf.keras.Model(inputs = quantized, outputs = results, **kwargs);

def VQVAE_Trainer(in_channels = 3, hidden_channels = 128, res_layer = 2, res_channels = 32, embed_dim = 128, n_embed = 10000, quantize_type = 'original'):
  inputs = tf.keras.Input((None, None, in_channels));
  quantized, loss = VQVAE_Encoder(in_channels, hidden_channels, res_layer, res_channels, embed_dim, n_embed, quantize_type)(inputs);
  recon = VQVAE_Decoder(in_channels, hidden_channels, res_layer, res_channels, embed_dim)(quantized);
  return tf.keras.Model(inputs = inputs, outputs = (recon, loss));

if __name__ == "__main__":
  import numpy as np;
  trainer = VQVAE_Trainer();
  trainer.save('trainer.h5');
  inputs = np.random.normal(size = (4,64,64,3));
  recon, loss = trainer(inputs);
  print(recon.shape, loss.shape);
  encoder = VQVAE_Encoder();
  encoder.save('encoder.h5');
  quantized, loss = encoder(inputs);
  print(quantized.shape, loss.shape);
