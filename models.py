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
  inputs = tf.keras.Input((None, None, input_channels)); # inputs.shape = (batch, height, width, input_channels)
  results = tf.keras.layers.Conv2D(hidden_channels, (7,7), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(input_channels * 7 ** 2)))(inputs); # results.shape = results.shape = (batch, height, width, hidden_channels)
  for i in range(group_num):
    ic = max(2**(i-1) * hidden_channels, hidden_channels);
    oc = 2**i * hidden_channels;
    for j in range(blk_per_group):
      results = EncoderBlock(ic if j == 0 else oc, oc, blk_per_group * group_num)(results);
    if i != group_num - 1:
      results = tf.keras.layers.MaxPool2D(pool_size = (2, 2))(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(vocab_size, (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(2**(group_num - 1) * 1 ** 2)))(results);
  results = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1))(results); # results.shape = (batch, height / 2 ** (group_num - 1), width / 2 ** (group_num - 1))
  results = tf.keras.layers.Lambda(lambda x, v: tf.one_hot(x, v), arguments = {'v': vocab_size})(results);
  # NOTE: results.shape = (batch, height / 2 ** (group_num - 1), width / 2 ** (group_num - 1), vocab_size)
  return tf.keras.Model(inputs = inputs, outputs = results);

def DecoderBlock(input_channels, output_channels, total_layers):
  hidden_channels = output_channels // 4;
  inputs = tf.keras.Input((None, None, input_channels)); # inputs.shape = (batch, height, width, input_channels)
  results = tf.keras.layers.ReLU()(inputs);
  results = tf.keras.layers.Conv2D(hidden_channels, (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(input_channels * 1 ** 2)))(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(hidden_channels, (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(hidden_channels * 3 ** 2)))(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(hidden_channels, (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(hidden_channels * 3 ** 2)))(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(output_channels, (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(hidden_channels * 3 ** 2)))(results);
  if input_channels != output_channels:
    short = tf.keras.layers.Conv2D(output_channels, (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(input_channels * 1 ** 2)))(inputs);
  else:
    short = inputs;
  results = tf.keras.layers.Lambda(lambda x, l: x[0] * 1 / (l ** 2) + x[1], arguments = {'l': total_layers})([results, short]);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Decoder(img_channels = 3, init_channels = 128, hidden_channels = 256, blk_per_group = 2, group_num = 4, vocab_size = 8192):
  assert img_channels >= 1;
  assert init_channels >= 8;
  assert hidden_channels >= 64;
  assert blk_per_group >= 1;
  assert vocab_size >= 512;
  inputs = tf.keras.Input((None, None, vocab_size)); # inputs.shape = (batch, height, width, input_channels)
  results = tf.keras.layers.Conv2D(init_channels, (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(vocab_size * 1 ** 2)))(inputs); # results.shape = results.shape = (batch, height, width, hidden_channels)
  for i in range(group_num - 1, -1, -1):
    ic = 2**i * hidden_channels;
    oc = max(2**(i-1) * hidden_channels, hidden_channels);
    for j in range(blk_per_group):
      results = DecoderBlock(init_channels if i == group_num - 1 and j == 0 else (ic if j == 0 else oc), oc, blk_per_group * group_num)(results);
    if i != 0:
      results = tf.keras.layers.UpSampling2D(size = (2, 2), interpolation = 'nearest')(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(2 * img_channels, (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1/sqrt(2**(group_num - 1) * 1 ** 2)))(results);
  results = tf.keras.layers.Lambda(lambda x, c: tf.math.sigmoid(x[...,:c]), arguments = {'c': img_channels})(results);
  # NOTE: results.shape = (batch, height * 2 ** (group_num - 1), width * 2 ** (group_num - 1), img_channels)
  return tf.keras.Model(inputs = inputs, outputs = results);

def Preprocess(target_image_size = 256, logit_laplace_eps = 0.1):
  # NOTE: height and width must be greater than or equal to target_image_size
  inputs = tf.keras.Input((None, None, 3)); # inputs.shape = (batch, height, width, 3)
  float_inputs = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype = tf.float32))(inputs);
  smaller_side = tf.keras.layers.Lambda(lambda x: tf.math.reduce_min(tf.shape(x)[1:3]))(float_inputs); # smaller_side.shape = ()
  scale = tf.keras.layers.Lambda(lambda x, t: t / x, arguments = {'t': target_image_size})(smaller_side); # scale.shape = ()
  size = tf.keras.layers.Lambda(lambda x: tf.cast(tf.cast(tf.shape(x[0])[1:3], dtype = tf.float32) * x[1], dtype = tf.int32))([float_inputs, scale]); # size.shape = (2,)
  results = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], x[1], tf.image.ResizeMethod.LANCZOS5))([float_inputs, size]); # results.shape = (batch, height * scale, width * scale, 3)
  # NOTE: the smaller side equals to target_image_size
  results = tf.keras.layers.Lambda(lambda x, t: tf.pad(x, [[0,0],[t//2, t - t//2],[t//2, t - t//2],[0,0]]), arguments = {'t': target_image_size})(results); # results.shape = (batch, 256 + height * scale, 256 + width * scale, 3)
  results = tf.keras.layers.Lambda(lambda x, t: tf.cond(tf.math.less(tf.shape(x)[1], tf.shape(x)[2]), 
                                                        lambda: x[:,:,(tf.shape(x)[2] - 2*t)//2:(tf.shape(x)[2] - 2*t)//2+2*t,:],
                                                        lambda: x[:,(tf.shape(x)[1] - 2*t)//2:(tf.shape(x)[1] - 2*t)//2+2*t,:,:]), arguments = {'t': target_image_size})(results); # results.shape = (batch, 2*target_image_size, 2*target_image_size, 3)
  results = tf.keras.layers.Lambda(lambda x, l: (1 - 2 * l) * x + l, arguments = {'l': logit_laplace_eps})(results);
  # NOTE: results.shape = (batch, 2 * target_image_size, 2 * target_image_size, 3)
  return tf.keras.Model(inputs = inputs, outputs = results);

def Postprocess(logit_laplace_eps = 0.1):
  inputs = tf.keras.Input((None, None, 3));
  results = tf.keras.layers.Lambda(lambda x, l: (x - l) / (1 - 2 * l), arguments = {'l': logit_laplace_eps})(inputs);
  results = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 1))(results);
  results = tf.keras.layers.Lambda(lambda x: tf.cast(255 * x, dtype = tf.uint8))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  import numpy as np;
  preprocess = Preprocess();
  postprocess = Postprocess();
  encoder = Encoder(group_num = 4);
  decoder = Decoder(group_num = 4);
  inputs = np.random.normal(size = (4, 360, 640, 3));
  results = preprocess(inputs); print(results.shape);
  results = encoder(results); print(results.shape);
  results = decoder(results); print(results.shape);
  results = postprocess(results); print(results.shape);

  preprocess.save('preprocess.h5');
  postprocess.save('postprocess.h5');
  encoder.save('encoder.h5');
  decoder.save('decoder.h5');
