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
  # 3) split the dimension to form multiple num_heads
  query_splitted = tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads))(query_dense);
  query_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(query_splitted); # query_splitted.shape = (batch, heads, input_length, dimension // heads)
  key_splitted = tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads))(key_dense);
  key_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(key_splitted); # key_splitted.shape = (batch, heads, key_length, dimension // heads)
  value_splitted = tf.keras.layers.Reshape((01, num_heads, d_model // num_heads))(value_dense);
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
