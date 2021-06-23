#!/usr/bin/python3

import tensorflow as tf;

class Quantize(tf.keras.layers.Layer):
  def __init__(self, dim, n_embed, decay = 0.99, eps = 1e-5, **kwargs):
    self.dim = dim;
    self.n_embed = n_embed;
    self.decay = decay;
    self.eps = eps;
    super(Quantize, self).__init__(**kwargs);
  def build(self, input_shape):
    # NOTE: all this weights are not trainable, because I manage them manually
    # cluster_mean: cluster_meanding code book
    # cluster_size: how many samples falling in each cluster
    # cluster_sum: the respective sum of samples falling in each cluster
    self.cluster_mean = self.add_weight(shape = (self.dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = False, name = 'cluster_mean');
    self.cluster_size = self.add_weight(shape = (self.n_embed,), dtype = tf.float32, initializer = tf.keras.initializers.Zeros(), trainable = False, name = 'cluster_size');
    self.cluster_sum = self.add_weight(shape = (self.dim, self.n_embed), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(stddev = 1.), trainable = False, name = 'cluster_mean');
    self.cluster_sum.assign(self.cluster_mean);
  def call(self, inputs):
    samples = tf.keras.layers.Reshape((self.dim,))(inputs); # samples.shape = (n_sample, dim)
    # dist = (X - cluster_mean)^2 = X' * X - 2 * X' * Embed + trace(Embed' * Embed),  dist.shape = (n_sample, n_embed), euler distances to cluster_meanding vectors
    dist = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.pow(x[0],2), axis = 1, keepdims = True) - 2 * tf.linalg.matmul(x[0], x[1]) + tf.math.reduce_sum(tf.math.pow(x[1],2), axis = 0, keepdims = True))([samples, self.cluster_mean]);
    cluster_index = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = 1))(dist); # cluster_index.shape = (n_sample)
    quantize = tf.keras.layers.Lambda(lambda x: tf.nn.embedding_lookup(tf.transpose(x[0]), x[1]))([self.cluster_mean, cluster_index]); # quantize.shape = (n_sample, dim)
    diff = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.math.pow(x[0] - x[1], 2), axis = -1))([inputs, quantize]); # diff.shape = (n_sample,)
    if tf.keras.backend.learning_phase() == 1:
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
    return quantize, cluster_index, diff;
  def get_config(self):
    config = super(Quantize, self).get_config();
    config['dim'] = self.dim;
    config['n_embed'] = self.n_embed;
    config['decay'] = self.decay;
    config['eps'] = self.eps;
  @classmethod
  def from_config(cls, config):
    self.dim = config['dim'];
    self.n_embed = config['n_embed'];
    self.decay = config['decay'];
    self.eps = config['eps'];
    return cls(**config);
    

def Encoder(img_channels = 3, hidden_channels = 128, vocab_size = 10000):
  inputs = tf.keras.Input((None, None, img_channels));
  results = tf.keras.layers.Conv2D(hidden_channels, (4,4), strides = (2,2), padding = 'same')(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(hidden_channels, (4,4), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(vocab_size, (1,1), padding = 'same')(results);
  tokens = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1))(results);
  return tf.keras.Model(inputs = inputs, outputs = tokens);

def Decoder(img_channels = 3, hidden_channels = 128, vocab_size = 10000):
  tokens = tf.keras.Input((None, None, 1), dtype = tf.int32); # inputs.shape = (batch, 16, 16, 1)
  results = tf.keras.layers.Lambda(lambda x, v: tf.one_hot(tf.squeeze(x, -1), v), arguments = {'v': vocab_size})(tokens);
  cluster_meandings = tf.keras.layers.Conv2D(hidden_channels, (1,1), padding = 'same')(results); # cluster_meandings.shape = (batch, 16, 16, hidden_channels)
  results = tf.keras.layers.Conv2DTranspose(hidden_channels, (4,4), strides = (2,2), padding = 'same')(cluster_meandings);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2DTranspose(hidden_channels, (4,4), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(img_channels, (1,1), padding = 'same')(results);
  return tf.keras.Model(inputs = tokens, outputs = results);

def MultiHeadAttention(d_model = 1024, length = 35, height = 16, width = 16, num_heads = 8):
  # 1) inputs
  query = tf.keras.Input((length * height * width, d_model)); # query.shape = (batch, (encode|decode)_length * (encode|decode)_height * (encode|decode)_width, d_model)
  key = tf.keras.Input((length * height * width, d_model)); # key.shape = (batch, encode_length * encode_height * encode_width, d_model)
  value = tf.keras.Input((length * height * width, d_model)); # value.shape = (batch, encode_length * encode_height * encode_width, d_model)
  t_mask = tf.keras.Input((1, None, None)); # t_mask.shape = (batch, 1, input_length = decode_length, key_length = (encode|decode)_length)
  r_mask = tf.keras.Input((1, None, None)); # r_mask.shape = (batch, 1, input_length = decode_height, key_length = (encode|decode)_height)
  c_mask = tf.keras.Input((1, None, None)); # c_mask.shape = (batch, 1, input_length = decoce_width, key_length = (encode|decode)_width)
  # 2) encoding
  query_dense = tf.keras.layers.Dense(units = d_model)(query);
  key_dense = tf.keras.layers.Dense(units = d_model)(key);
  value_dense = tf.keras.layers.Dense(units = d_model)(value);
  # 3) split the dimension to form multiple num_heads
  query_splitted = tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads))(query_dense);
  query_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(query_splitted); # query_splitted.shape = (batch, heads, input_length, dimension // heads)
  key_splitted = tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads))(key_dense);
  key_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(key_splitted); # key_splitted.shape = (batch, heads, key_length, dimension // heads)
  value_splitted = tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads))(value_dense);
  value_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(value_splitted); # value_splitted.shape = (batch, heads, value_length, dimension // heads)
  # 4) weighted sum of value elements for each query element
  SA_t = Attention(d_model * height * width, num_heads);
  SA_r = Attention(d_model * length * width, num_heads);
  SA_c = Attention(d_model * length * height, num_heads);
  t_key_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (tf.shape(x)[0], tf.shape(x)[1], t, r * c * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(key_splitted); # t_key_splitted.shape = (batch, heads, key_length / (height * width), (height * width) * dimension // heads)
  t_value_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (tf.shape(x)[0], tf.shape(x)[1], t, r * c * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(value_splitted); # t_value_splitted.shape = (batch, heads, value_length / (height * width), (height * width) * dimension // heads)
  r_key_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (0, 1, 3, 2, 4, 5)), (tf.shape(x)[0], tf.shape(x)[1], r, t * c * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(key_splitted); # r_key_splitted.shape = (batch, heads, key_length / (length * width), (length * width) * dimension // heads)
  r_value_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (0, 1, 3, 2, 4, 5)), (tf.shape(x)[0], tf.shape(x)[1], r, t * c * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(value_splitted); # r_value_splitted.shape = (batch, heads, value_length / (length * width), (length * width) * dimension // heads)
  c_key_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (0, 1, 4, 2, 3, 5)), (tf.shape(x)[0], tf.shape(x)[1], c, t * r * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(key_splitted); # c_key_splitted.shape = (batch, heads, key_length / (length * height), (length * height) * dimension // heads)
  c_value_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (0, 1, 4, 2, 3, 5)), (tf.shape(x)[0], tf.shape(x)[1], c, t * r * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(value_splitted); # c_value_splitted.shape = (batch, heads, value_length / (length * height), (length * height) * dimension // heads)
  for i in range(4):
    t_query_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (tf.shape(x)[0], tf.shape(x)[1], t, r * c * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(query_splitted); # t_query_splitted.shape = (batch, heads, input_length / (height * width), (height * width) * dimension // heads)
    t_query_splitted = SA_t([t_query_splitted, t_key_splitted, t_value_splitted, t_mask]); # t_query_splitted.shape = (batch, heads, input_length / (height * width), (height * width) * dimension // heads)
    query_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3] / (r * c))), (tf.shape(x)[0], tf.shape(x)[1], t * r * c, tf.shape(x)[3] / (r * c))), arguments = {'t': length,'r': height,'c': width})(t_query_splitted); # query_splitted.shape = (batch, heads, input_length, dimension // heads)
    r_query_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (0, 1, 3, 2, 4, 5)), (tf.shape(x)[0], tf.shape(x)[1], r, t * c * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(query_splitted); # r_query_splitted.shape = (batch, heads, input_length / (length * width), (length * width) * dimension // heads)
    r_query_splitted = SA_r([r_query_splitted, r_key_splitted, r_value_splitted, r_mask]); # r_query_splitted.shape = (batch, heads, input_length / (length * width), (length * width) * dimension // heads)
    query_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], r, t, c, tf.shape(x)[3] / (t * c))), (0, 1, 3, 2, 4, 5)), (tf.shape(x)[0], tf.shape(x)[1], t * r * c, tf.shape(x)[3] / (t * c))), arguments = {'t': length,'r': height,'c': width})(r_query_splitted); # query_splitted.shape = (batch, heads, input_length, dimension // heads)
    c_query_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], t, r, c, tf.shape(x)[3])), (0, 1, 4, 2, 3, 5)), (tf.shape(x)[0], tf.shape(x)[1], c, t * r * tf.shape(x)[3])), arguments = {'t': length,'r': height,'c': width})(query_splitted); # c_query_splitted.shape = (batch, heads, input_length / (length * height), (length * height) * dimension // heads)
    c_query_splitted = SA_c([c_query_splitted, c_key_splitted, c_value_splitted, c_mask]); # c_query_splitted.shape = (batch, heads, input_length / (length * height), (length * height) * dimension // heads)
    query_splitted = tf.keras.layers.Lambda(lambda x, t, r, c: tf.reshape(tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], c, t, r, tf.shape(x)[3] / (t * r))), (0, 1, 3, 4, 2, 5)), (tf.shape(x)[0], tf.shape(x)[1], t * r * c, tf.shape(x)[3] / (t * r))), arguments = {'t': length,'r': height,'c': width})(c_query_splitted); # query_splitted.shape = (batch, heads, input_length, dimension // heads)
  # 5) concat heads
  attended = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(query_splitted); # attended.shape = (batch, input_length, heads, dimension // heads)
  concated = tf.keras.layers.Reshape((-1, d_model))(attended); # concated.shape = (batch, input_length, dimension)
  # 6) output
  results = tf.keras.layers.Dense(units = d_model)(concated); # results.shape = (batch, input_length, dimension)
  return tf.keras.Model(inputs = (query, key, value, t_mask, r_mask, c_mask), outputs = results);

if __name__ == "__main__":
  import numpy as np;
  encoder = Encoder();
  decoder = Decoder();
  inputs = np.random.normal(size = (4, 64,64,3));
  tokens = encoder(inputs);
  print(tokens.shape);
  results = decoder(tokens);
  print(results.shape);
  encoder.save('encoder.h5');
  decoder.save('decoder.h5');
  q = Quantize(10,5);
  inputs = np.random.normal(size = (4,10));
  tf.keras.backend.set_learning_phase(1);
  outputs = q(inputs);
