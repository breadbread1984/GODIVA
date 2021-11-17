#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;
from vqvae import *;

def Attention(d_model, num_heads):

    # NOTE: mask = 1 means ignoring the self attention element.
    # dimension must be divisible by num_heads
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # 1) inputs
    query = tf.keras.Input((num_heads, None, d_model // num_heads)); # query.shape = (batch, heads, input_length, dimension)
    key = tf.keras.Input((num_heads, None, d_model // num_heads));   # key.shape = (batch, heads, key_length = value_length, dimension)
    value = tf.keras.Input((num_heads, None, d_model // num_heads)); # value.shape = (batch, heads, value_length, dimension)
    mask = tf.keras.Input((1, None, None));                          # mask.shape = (batch, 1, input_length or 1, key_length)
    # 2) self attention weight for each query element
    qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([query, key]); # qk.shape = (batch, heads, input_length, key_length)
    depth = tf.keras.layers.Lambda(lambda x: tf.cast(tf.shape(x)[-1], dtype = tf.float32))(key);           # depth = dimension
    # NOTE: mask is propagated here!
    logits = tf.keras.layers.Lambda(lambda x: x[0] / tf.math.sqrt(x[1]) + (1 - x[2]) * -1e9)([qk, depth, mask]); # logits.shape = (batch, heads, input_length, key_length)
    attention = tf.keras.layers.Softmax()(logits);                                                         # attention.shape = (batch, heads, input_length, key_length)
    # 3) weighted sum of value elements for each query element
    results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, value]); # results.shape = (batch, heads, input_length, dimension)
    return tf.keras.Model(inputs = (query, key, value, mask), outputs = results);

def MultiHeadAttention(d_model, num_heads):
    
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # 1) inputs
    query = tf.keras.Input((None,d_model)); # query.shape = (batch, input_length, dimension)
    key = tf.keras.Input((None,d_model));   # key.shape = (batch, key_length = value_length, dimension)
    value = tf.keras.Input((None,d_model)); # value.shape = (batch, value_length, dimension)
    mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, input_length or 1, key_length)
    # 2) encoding
    query_dense = tf.keras.layers.Dense(units = d_model)(query); # query_dense.shape = (batch, input_length, dimension)
    key_dense = tf.keras.layers.Dense(units = d_model)(key);     # key_dense.shape = (batch, key_length, dimension)
    value_dense = tf.keras.layers.Dense(units = d_model)(value); # value_dense.shape = (batch, value_length, dimension)
    # 3) split the dimension to form multiple heads
    query_splitted = tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads))(query_dense);
    query_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(query_splitted); # query_splitted.shape = (batch, heads, input_length, dimension // heads)
    key_splitted = tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads))(key_dense);
    key_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(key_splitted);     # key_splitted.shape = (batch, heads, key_length, dimension // heads)
    value_splitted = tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads))(value_dense);
    value_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(value_splitted); # query_splitted.shape = (batch, heads, value_length, dimension // heads)
    # 4) weighted sum of value elements for each query element
    attended = Attention(d_model, num_heads)([query_splitted, key_splitted, value_splitted, mask]);   # attended.shape = (batch, heads, input_length, dimension // heads)
    # 5) concat heads
    attended = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(attended); # attended.shape = (batch, input_length, heads, dimension // heads)
    concated = tf.keras.layers.Reshape((-1, d_model))(attended);                          # concated.shape = (batch, input_length, dimension)
    # 6) output
    results = tf.keras.layers.Dense(units = d_model)(concated);                           # results.shape = (batch, input_length, dimension)
    return tf.keras.Model(inputs = (query, key, value, mask), outputs = results);

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

def EncoderLayer(d_model, num_heads, code_dim, dropout_rate, activation = 'relu'):

    assert activation in ['relu', 'gelu'];
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # 1) inputs
    inputs = tf.keras.Input((None, d_model));   # inputs.shape = (batch, encode_length, dimension)
    mask = tf.keras.Input((1, 1, None));        # mask.shape = (batch, 1, 1(will be encode_length), encode_length)
    # 2) multi-head attention resblock
    attended = MultiHeadAttention(d_model, num_heads)([inputs, inputs, inputs, mask]);
    attended = tf.keras.layers.Dropout(rate = dropout_rate)(attended);
    inputs_attended = tf.keras.layers.Add()([inputs, attended]);
    attended = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(inputs_attended); # attended.shape = (batch, encode_length, dimension)
    # 3) feed forward network resblock
    outputs = tf.keras.layers.Dense(units = code_dim)(attended);
    if activation == 'gelu':
      outputs = tfa.layers.GELU()(outputs);
    else:
      outputs = tf.keras.layers.ReLU()(outputs);
    outputs = tf.keras.layers.Dense(units = d_model)(outputs);
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(outputs);
    attended_outputs = tf.keras.layers.Add()([attended, outputs]);                  # attended_outputs.shape = (batch, encode_length, dimension)
    outputs = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(attended_outputs); # outputs.shape = (batch, encode_length, dimension)
    return tf.keras.Model(inputs = (inputs, mask), outputs = outputs);

def Encoder(vocab_size, num_layers, d_model, num_heads, code_dim, dropout_rate, activation = "relu"):

    assert activation in ['relu', 'gelu'];
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # 1) inputs
    inputs = tf.keras.Input((None,));        # inputs.shape = (batch, encode_length)
    mask = tf.keras.Input((1, 1, None));  # mask.shape = (batch, 1, 1(will be encode_length), encode_length)
    # 2) token to positional embedding
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs);
    embeddings = tf.keras.layers.Lambda(lambda x, d_model: tf.math.sqrt(tf.cast(d_model, dtype = tf.float32)) * x, arguments = {'d_model': d_model})(embeddings);
    embeddings = PositionalEncoding(d_model)(embeddings);
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(embeddings);  # embeddings.shape = (batch, encode_length, dimension)
    # 3) multiple encode layers
    for i in range(num_layers):
        outputs = EncoderLayer(d_model, num_heads, code_dim, dropout_rate, activation)([outputs, mask]); # outputs.shape = (batch, encode_length, dimension)
    return tf.keras.Model(inputs = (inputs, mask), outputs = outputs);

def DecoderLayer(d_model, num_heads, code_dim, dropout_rate, activation = "relu"):
    
    assert activation in ['relu', 'gelu'];
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # 1) inputs
    inputs = tf.keras.Input((None, d_model));          # inputs.shape = (batch, decode_length, dimension)
    code = tf.keras.Input((None, d_model));            # code.shape = (batch, encode_length, dimension)
    look_ahead_mask = tf.keras.Input((1, None, None)); # look_ahead_mask.shape = (batch, 1, decode_length, encode_length)
    padding_mask = tf.keras.Input((1, 1, None));       # padding_mask.shape = (batch, 1, 1(will be decode_length), encode_length)
    # 2) multi-head attention resblock
    attention1 = MultiHeadAttention(d_model, num_heads)([inputs, inputs, inputs, look_ahead_mask]);
    attention1_inputs = tf.keras.layers.Add()([attention1, inputs]);
    attention1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(attention1_inputs);     # attention1.shape = (batch, decode_length, dimension)
    # 3) multi-head attention
    attention2 = MultiHeadAttention(d_model, num_heads)([attention1, code, code, padding_mask]);
    attention2 = tf.keras.layers.Dropout(rate = dropout_rate)(attention2);
    attention2_attention1 = tf.keras.layers.Add()([attention2, attention1]);
    attention2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(attention2_attention1); # attention2.shape = (batch, decode_length, dimension)
    # 4) feed ward network
    outputs = tf.keras.layers.Dense(units = code_dim)(attention2);
    if activation == 'gelu':
      outputs = tfa.layers.GELU()(outputs);
    else:
      outputs = tf.keras.layers.ReLU()(outputs);
    outputs = tf.keras.layers.Dense(units = d_model)(outputs);
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(outputs);
    outputs_attention2 = tf.keras.layers.Add()([outputs, attention2]);                      # outputs_attention2.shape = (batch, decode_length, dimension)
    outputs = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(outputs_attention2);       # outputs.shape = (batch, decode_length, dimension)
    # 5) outputs.shape = (batch, seq_length, d_model)
    return tf.keras.Model(inputs = (inputs, code, look_ahead_mask, padding_mask), outputs = outputs);

def Decoder(vocab_size, num_layers, d_model, num_heads, code_dim, dropout_rate, activation = 'relu'):
    
    assert activation in ['relu', 'gelu'];
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # 1) inputs
    inputs = tf.keras.Input((None,));                  # inputs.shape = (batch, decode_length)
    code = tf.keras.Input((None, d_model));            # code.shape = (batch, encode_length, dimension)
    look_ahead_mask = tf.keras.Input((1, None, None)); # look_ahead_mask.shape = (batch, 1, decode_length, encode_length)
    padding_mask = tf.keras.Input((1, 1, None));       # padding_mask.shape = (batch, 1, 1(will be decode_length), encode_length)
    # 2) token to positional embedding
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs);
    embeddings = tf.keras.layers.Lambda(lambda x, d_model: tf.math.sqrt(tf.cast(d_model, dtype = tf.float32)) * x, arguments = {'d_model': d_model})(embeddings);
    embeddings = PositionalEncoding(d_model)(embeddings);
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(embeddings); # outputs.shape = (batch, decode_length, dimension)
    # 3) multiple decode layers
    for i in range(num_layers):
        outputs = DecoderLayer(d_model, num_heads, code_dim, dropout_rate, activation)([outputs, code, look_ahead_mask, padding_mask]); # outputs.shape = (batch, decode_length, dimension)
    return tf.keras.Model(inputs = (inputs, code, look_ahead_mask, padding_mask), outputs = outputs);

def Transformer(enc_vocab_size, dec_vocab_size, num_layers = 2, d_model = 256, num_heads = 8, code_dim = 512, dropout_rate = 0.1, activation = 'relu'):
    
    assert activation in ['relu', 'gelu'];

    def create_look_ahead_mask(x):
        # self attention mask: every token only related to tokens before it
        seq_len = tf.shape(x)[1]; # input_length
        # set upper triangular zero
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0);
        padding_mask = tf.cast(tf.math.equal(x, 0), tf.float32);
        padding_mask = tf.expand_dims(tf.expand_dims(padding_mask, 1), 1);
        return tf.maximum(look_ahead_mask, padding_mask); # union the mask

    # 1) inputs
    enc_inputs = tf.keras.Input((None,));                                         # inputs.shape = (batch, encode_length)
    dec_inputs = tf.keras.Input((None,));                                         # dec_inputs.shape = (batch, decode_length)
    enc_padding_mask = tf.keras.Input((1,1,None));                                # enc_padding_mask.shape = (batch, 1, 1(will be encode_length), encode_length)
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask)(dec_inputs); # look_ahead_mask.shape = (batch, 1, decode_length, encode_length)
    # 2) generate code
    code = Encoder(enc_vocab_size, num_layers, d_model, num_heads, code_dim, dropout_rate, activation)([enc_inputs, enc_padding_mask]); # code.shape = (batch, encode_length, dimension)
    decoded = Decoder(dec_vocab_size, num_layers, d_model, num_heads, code_dim, dropout_rate, activation)([dec_inputs, code, look_ahead_mask, enc_padding_mask]); # decoded.shape = (batch, decode_length, dimension)
    # 3) output
    outputs = tf.keras.layers.Dense(units = dec_vocab_size)(decoded); # outputs.shape = (batch, decode_length, vocab_size)
    return tf.keras.Model(inputs = (enc_inputs, enc_padding_mask, dec_inputs), outputs = outputs);

class GODIVA(tf.keras.Model):
  def __init__(self, img_size = 64, video_length = 16, text_vocab_size = None, video_vocab_size = 10000, **kwargs):
    super(GODIVA, self).__init__(**kwargs);
    self.video_length = video_length;
    self.transformer = Transformer(text_vocab_size + 2, video_vocab_size + 2);
    self.VIDEO_SOS = video_vocab_size;
    self.VIDEO_EOS = video_vocab_size + 1;
    self.frame_token_num = img_size // 4 * img_size // 4;
  def call(self, inputs):
    # inputs.shape = (batch, length)
    text = inputs[0]; # text.shape = (batch, text_length)
    mask = inputs[1]; # mask.shape = (batch, 1, 1, text_length)
    total_tokens = tf.ones((tf.shape(text)[0], 1), dtype = tf.int64) * self.VIDEO_SOS; # tokens.shape = (batch, 1)
    total_preds = tf.ones((tf.shape(text)[0], 0, self.transformer.output[0].shape[-1]), dtype = tf.float32); # preds.shape = (batch, 0, video_vocab_size + 2)
    for i in range(self.video_length * self.frame_token_num + 1):
      pred = self.transformer([text, mask, total_tokens]); # bottom_pred.shape = (batch, length, video_vocab_size + 2)
      tokens = tf.math.argmax(pred, axis = -1); # tokens.shape = (batch, length)
      total_tokens = tf.concat([total_tokens, tokens[:,-1:]], axis = 1); # bottom_tokens.shape = (batch, length + 1)
      total_preds = tf.concat([total_preds, pred[:,-1:,:]], axis = 1); # bottom_tokens.shape = (batch, length + 1, video_vocab_size + 2)
    # NOTE: total_preds.shape = (batch, video_length * frame_token_num + 2, video_vocab_size + 2)
    return total_preds;

if __name__ == "__main__":

  tokens = np.random.randint(low = 0, high = 10, size = (1, 34));
  mask = np.random.randint(low = 0, high = 2, size = (1, 1, 1, 34));
  godiva = GODIVA(text_vocab_size = 10);
  preds = godiva([tokens, mask]);
  print(preds.shape);
  godiva.save_weights('godiva.h5');
