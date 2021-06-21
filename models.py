#!/usr/bin/python3

from math import sqrt;
import tensorflow as tf;

def EncoderBlock(input_channels, output_channels, total_layers):
  hidden_channels = output_channels // 4;
  inputs = tf.keras.Input((None, None, input_channels)); # inputs.shape = (batch, height, width, input_channels)
  results = tf.keras.layers.ReLU()(inputs);
  results = tf.keras.layers.Conv2D(hidden_channels, (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(input_channels * 3 ** 2)))(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(hidden_channels, (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(hidden_channels * 3 ** 2)))(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(hidden_channels, (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(hidden_channels * 3 ** 2)))(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(output_channels, (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(hidden_channels * 1 ** 2)))(results);
  if input_channels != output_channels:
    short = tf.keras.layers.Conv2D(output_channels, (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(input_channels * 1 ** 2)))(inputs);
  else:
    short = inputs;
  results = tf.keras.layers.Lambda(lambda x, l: x[0] * 1 / (l ** 2) + x[1], arguments = {'l': total_layers})([results, short]);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Encoder(input_channels = 3, hidden_channels = 256, blk_per_group = 2, group_num = 4, vocab_size = 8192):
  assert input_channels >= 1;
  assert hidden_channels >= 64;
  assert blk_per_group >= 1;
  assert vocab_size >= 512;
  inputs = tf.keras.Input((2**group_num, 2**group_num, input_channels)); # inputs.shape = (batch, height, width, input_channels)
  results = tf.keras.layers.Conv2D(hidden_channels, (7,7), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(input_channels * 7 ** 2)))(inputs); # results.shape = results.shape = (batch, height, width, hidden_channels)
  for i in range(group_num):
    ic = max(2**(i-1) * hidden_channels, hidden_channels);
    oc = 2**i * hidden_channels;
    for j in range(blk_per_group):
      results = EncoderBlock(ic if j == 0 else oc, oc, blk_per_group * group_num)(results);
      if j != blk_per_group - 1:
        results = tf.keras.layers.MaxPool2D(pool_size = (2, 2))(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(vocab_size, (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(2**(group_num - 1) * 1 ** 2)))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  import numpy as np;
  encoder = Encoder(group_num = 4);
  encoder.save('encoder.h5');
  inputs = np.random.normal(size = (4, 16, 16, 3));
  outputs = encoder(inputs);
  print(outputs.shape);
