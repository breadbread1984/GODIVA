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
  results = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1))(results);
  results = tf.keras.layers.Lambda(lambda x, v: tf.one_hot(x, v), arguments = {'v': vocab_size})(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Decoder(img_channels = 3, hidden_channels = 128, vocab_size = 10000):
  inputs = tf.keras.Input((None, None, vocab_size)); # inputs.shape = (batch, 16, 16, vocab_size)
  results = tf.keras.layers.Conv2D(hidden_channels, (1,1), padding = 'same')(inputs); # results.shape = (batch, 16, 16, hidden_channels)
  results = tf.keras.layers.Conv2DTranspose(hidden_channels, (4,4), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2DTranspose(hidden_channels, (4,4), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(img_channels, (1,1), padding = 'same')(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  import numpy as np;
  encoder = Encoder();
  decoder = Decoder();
  inputs = np.random.normal(size = (4, 64,64,3));
  results = encoder(inputs);
  print(results.shape);
  results = decoder(results);
  print(results.shape);
  encoder.save('encoder.h5');
  decoder.save('decoder.h5');
