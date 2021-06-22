#!/usr/bin/python3

import tensorflow as tf;

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
  embeddings = tf.keras.layers.Conv2D(hidden_channels, (1,1), padding = 'same')(results); # embeddings.shape = (batch, 16, 16, hidden_channels)
  results = tf.keras.layers.Conv2DTranspose(hidden_channels, (4,4), strides = (2,2), padding = 'same')(embeddings);
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
