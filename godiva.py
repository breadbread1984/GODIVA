#!/usr/bin/python3

from os.path import join;
import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;
from sparse import *;
from vqvae import *;

def FullAttention(key_dim, value_dim, num_heads, drop_rate = 0.2, causal = True, use_sparse_op = False):
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  if causal == True:
    y = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.shape(x)[2], dtype = tf.int32),(-1,1)),(1,tf.shape(x)[2])))(query);
    x = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.shape(x)[2], dtype = tf.int32),(1,-1)),(tf.shape(x)[2],1)))(query);
    mask = tf.keras.layers.Lambda(lambda x: tf.where(tf.math.greater_equal(x[0], x[1]), tf.ones_like(x[0]), tf.zeros_like(x[1])))([y,x]);
    mask = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(x[0], (1,1,tf.shape(x[0])[0],tf.shape(x[0])[1])), (tf.shape(x[1])[0],1,1,1)))([mask, query]);
  else:
    mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length, key_length)
  if use_sparse_op == True:
    qk = MaskedDenseMatMul()([query, key, mask]);
    logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads, query_length, key_length)
    logits = Dense2Sparse()([logits, mask]); # logits.shape = (batch, heads, query_length, key_length)
    attention = tf.keras.layers.Lambda(lambda x: tf.sparse.softmax(x))(logits); # attention.shape = (batch, num_heads, query_length, key_length)
    results = SparseDenseMatMul()([attention, value]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  else:
    qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([query, key]);
    logits = tf.keras.layers.Lambda(lambda x, kd: x[0] / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)) + tf.cast(1 - x[1], dtype = tf.float32) * -1e9, arguments = {'kd': key_dim // num_heads})([qk, mask]);
    attention = tf.keras.layers.Softmax()(logits);
    results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, value]);
  return tf.keras.Model(inputs = (query, key, value) if causal == True else (query, key, value, mask), outputs = results);

def AxialAttention(key_dim, value_dim, num_heads, drop_rate = 0.5, origin_shape = None, axial_dim = 0):
  # NOTE: this attention can only apply to self attention, but cross attention.
  # in other words, query_length = key_length must hold
  # NOTE: leave one dim as seq_length, merge the other dims with heads.
  # for example key.shape = (batch, heads, h, w, c, dim) and axial_dim = -2
  # key.shape becomes (batch, new_heads = heads * h * c, seq_length = w, dim),
  # the self attention matrix become w x w, rather than (h * w * c) x (h * w * c)
  assert type(origin_shape) in [list, tuple];
  assert 0 <= axial_dim < 3 or -3 <= axial_dim < 0;
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query.shape = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  reshaped_query = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], key_dim // num_heads))(query);
  reshaped_key = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], key_dim // num_heads))(key);
  reshaped_value = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], value_dim // num_heads))(value);
  def get_perm(axial_dim):
    dims = np.arange(2 + 3 + 1); # batch x heads x *origin_shape x dim
    chosed_dim = 2 + axial_dim if axial_dim >= 0 else 2 + 3 + axial_dim;
    index = dims.tolist().index(chosed_dim);
    dims[index], dims[-2] = dims[-2], dims[index];
    return dims;
  reshaped_query = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_query);
  reshaped_query = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_query); # query.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  reshaped_key = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_key);
  reshaped_key = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_key); # key.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  reshaped_value = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_value);
  shape = tf.keras.layers.Lambda(lambda x: tf.shape(x))(reshaped_value);
  reshaped_value = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_value); # value.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  # 1) correlation matrix of query and key
  qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([reshaped_query, reshaped_key]);
  logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  attention = tf.keras.layers.Softmax()(logits); # attention.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  attention = tf.keras.layers.Dropout(rate = drop_rate)(attention);
  # 2) weighted sum of value elements for each query element
  results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, reshaped_value]); # results.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, value_dim // heads)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], x[1]))([results, shape]); # results.shape = (batch, heads, *other_dims, axial_dim_length, value_dim // heads)
  def get_inv_perm(axial_dim):
    perm = get_perm(axial_dim);
    return np.argsort(perm);
  results = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_inv_perm(axial_dim)})(results); # results.shape = (batch, heads, *origin_shape, value_dim // heads)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1, tf.shape(x)[-1])))(results); # results.shape = (batch, heads, query_length = np.prod(origin_shape), value_dim // heads)
  return tf.keras.Model(inputs = (query, key, value), outputs = results);

def SparseAttention(key_dim, value_dim, num_heads, drop_rate = 0.5, origin_shape = None, causal = True, local_blocks = 3):
  assert type(origin_shape) in [list, tuple];
  # NOTE: local is the first mask for the strided head
  # NOTE: strided is the second mask for the strided head
  def idx_to_coord(idx):
    # convert idx in block_shape to coord
    shape = origin_shape;
    shape_cum = tuple(np.flip(np.cumprod(np.flip(np.array(shape)))[:-1])) + (1,);
    coord = list();
    for i in range(len(shape)):
      coord.append(idx // shape_cum[i]);
      idx %= shape_cum[i];
    return coord;
  def coord_to_idx(coord):
    # convert block_shape coord to idx
    shape = origin_shape;
    shape_cum = tuple(np.flip(np.cumprod(np.flip(np.array(shape)))[:-1])) + (1,);
    idx = 0;
    for i in range(len(shape)):
      idx += coord[i] * shape_cum[i];
    return idx;
  def make_layout():
    shape = origin_shape;
    num_blocks = np.prod(shape);
    layout = np.zeros((num_blocks, num_blocks), dtype = np.bool);
    # 1) first mask for the strided head
    for latter in range(num_blocks):
      for former in range(max(0, latter - local_blocks), (latter + 1 if causal == True else min(num_blocks, latter + local_blocks))):
        # 1) causal case: latter can only look formers backward in time as far as local_blocks number of blocks, can't look forward in time
        # 2) non causal case: latter can only look formers backward in time as far as local blocks number of blocks,
        #                     latter can only look forward in time as far as local blocks number of blocks
        layout[latter, former] = True;
    # 2) second mask for the strided head
    for latter in range(num_blocks):
      latter_coord = idx_to_coord(latter);
      # calculate all visible formers
      for d in range(len(shape) - 1):
        for i in range(0, (latter_coord[d]+1 if causal else shape[d])):
          # 1) causal case: latter can only look formers backward in time in every dimension of the block
          # 2) non causal case: latter can look forward and backward all elements in every dimension
          former_coord = latter_coord.copy();
          former_coord[d] = i;
          former = coord_to_idx(former_coord);
          if former >= num_blocks:
            print(former_coord);
          layout[latter, former] = True;
    layout = np.reshape(layout, (1,1,num_blocks,num_blocks)); # layout.shape = (1, 1, query_length, key_length)
    return layout;
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query.shape = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  mask = tf.keras.layers.Lambda(lambda x, m: tf.tile(m,(tf.shape(x)[0],1,1,1)), arguments = {'m': make_layout()})(query); # mask.shape = (batch, 1, query_length, key_length)
  qk = MaskedDenseMatMul()([query, key, mask]);
  logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads, query_length, key_length)
  logits = Dense2Sparse()([logits, mask]); # logits.shape = (batch, heads, query_length, key_length)
  attention = tf.keras.layers.Lambda(lambda x: tf.sparse.softmax(x))(logits); # attention.shape = (batch, num_heads, query_length, key_length)
  # FIXME: uncomment the following line
  #attention = SparseDropout(drop_rate = drop_rate)(attention); # attention.shape = (batch, num_heads, query_length, key_length)
  # 2) weighted sum of value elements for each query element
  results = SparseDenseMatMul()([attention, value]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  return tf.keras.Model(inputs = (query, key, value), outputs = results);

def MultiHeadAttention(key_dim, value_dim, num_heads, attn_type = 'full', **kwargs):
  assert attn_type in ['full', 'axial', 'sparse'];
  query = tf.keras.Input((None, key_dim,)); # query.shape = (batch, query_length, key_dim)
  key = tf.keras.Input((None, key_dim,)); # key.shape = (batch, key_length, key_dim)
  value = tf.keras.Input((None, value_dim,)); # value.shape = (batch, key_length, value_dim)
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
    if kwargs['causal'] == False:
      mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length, key_length)
    attended = FullAttention(key_dim, value_dim, num_heads, kwargs['drop_rate'], kwargs['causal'])([query_splitted, key_splitted, value_splitted] if kwargs['causal'] == True else [query_splitted, key_splitted, value_splitted, mask]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  elif attn_type == 'axial':
    attended = AxialAttention(key_dim, value_dim, num_heads, kwargs['drop_rate'], kwargs['origin_shape'], kwargs['axial_dim'])([query_splitted, key_splitted, value_splitted]); # reults.shape = (batch, num_heads, query_length, value_dim // num_heads)
  elif attn_type == 'sparse':
    attended = SparseAttention(key_dim, value_dim, num_heads, kwargs['drop_rate'], kwargs['origin_shape'], kwargs['causal'], kwargs['local_blocks'])([query_splitted, key_splitted, value_splitted]); # results.shape = (batch, num_heads, query_length, value_dim // num_heads)
  else:
    raise Exception('invalid attention type!');
  attended = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(attended); # attended.shape = (batch, query_length, num_heads, value_dim // num_heads)
  concated = tf.keras.layers.Reshape((-1, value_dim))(attended); # concated.shape = (batch, query_length, value_dim)
  # 3) output
  results = tf.keras.layers.Dense(key_dim)(concated); # results.shape = (batch, query_length, key_dim)
  return tf.keras.Model(inputs = (query, key, value) if attn_type != 'full' or kwargs['causal'] == True else (query, key, value, mask), outputs = results);

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

def EncoderLayer(hidden_dim = 1024, num_heads = 16, **kwargs):
  # NOTE: this sparse transformer residual block is for decoder only transformer
  inputs = tf.keras.Input((None, hidden_dim,)); # inputs.shape = (batch, hidden_length, hidden_dim)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length, key_length)
  short = inputs;
  results = tf.keras.layers.LayerNormalization()(inputs); # results.shape = (batch, hidden_length, hidden_dim)
  results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, attn_type = 'full', **kwargs)([results, results, results, mask]); # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results); # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Add()([results, short]); # results.shape = (batch, hidden_length, hidden_dim)
  short = results;
  results = tf.keras.layers.LayerNormalization()(results); # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Dense(hidden_dim * 4)(results); # results.shape = (batch, hidden_length, 4 * hidden_dim)
  results = tfa.layers.GELU()(results); # results.shape = (batch, hidden, 4 * hidden_dim)
  results = tf.keras.layers.Dense(hidden_dim)(results);  # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results); # results.shape = (batch, hidden_length, hidden_dim)
  results = tf.keras.layers.Add()([results, short]); # results.shape = (batch, hidden_length, hidden_dim)
  return tf.keras.Model(inputs = (inputs, mask), outputs = results);

def TransEncoder(num_layers = 2, hidden_dim = 1024, num_heads = 16, **kwargs):
  inputs = tf.keras.Input((None, hidden_dim));
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length, key_length)
  embeddings = PositionalEncoding(hidden_dim)(inputs);
  outputs = tf.keras.layers.Dropout(rate = kwargs['drop_rate'])(embeddings);
  for i in range(num_layers):
    outputs = EncoderLayer(hidden_dim, num_heads, drop_rate = kwargs['drop_rate'], causal = False)([outputs, mask]);
  return tf.keras.Model(inputs = (inputs, mask), outputs = outputs);

def DecoderLayer(hidden_dim = 1024, num_heads = 16, **kwargs):
  inputs = tf.keras.Input((None, hidden_dim,)); # inputs.shape = (batch, video_length, hidden_dim)
  code = tf.keras.Input((None, hidden_dim)); # code.shape = (batch, text_length, hidden_dim)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, video_length, text_length)
  short = inputs;
  results = tf.keras.layers.LayerNormalization()(inputs);
  for i in range(4):
    results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, attn_type = 'axial', drop_rate = kwargs['drop_rate'], origin_shape = kwargs['origin_shape'], axial_dim = -3)([results, results, results]);
    results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, attn_type = 'axial', drop_rate = kwargs['drop_rate'], origin_shape = kwargs['origin_shape'], axial_dim = -2)([results, results, results]);
    results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, attn_type = 'axial', drop_rate = kwargs['drop_rate'], origin_shape = kwargs['origin_shape'], axial_dim = -1)([results, results, results]);
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results);
  results = tf.keras.layers.Add()([results, short]);
  short = results;
  results = tf.keras.layers.LayerNormalization()(results);
  results = MultiHeadAttention(hidden_dim, hidden_dim, num_heads, 'full', drop_rate = kwargs['drop_rate'], causal = False)([results, code, code, mask]);
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results);
  results = tf.keras.layers.Add()([results, short]);
  short = results;
  results = tf.keras.layers.LayerNormalization()(results);
  results = tf.keras.layers.Dense(hidden_dim * 4)(results);
  results = tfa.layers.GELU()(results);
  results = tf.keras.layers.Dense(hidden_dim)(results);
  results = tf.keras.layers.Dropout(kwargs['drop_rate'])(results);
  results = tf.keras.layers.Add()([results, short]);
  return tf.keras.Model(inputs = (inputs, code, mask), outputs = results);

def TransDecoder(num_layers = 2, hidden_dim = 1024, num_heads = 16, **kwargs):
  inputs = tf.keras.Input((None, hidden_dim)); # inputs.shape = (batch, video_length, hidden_dim)
  code = tf.keras.Input((None, hidden_dim)); # code.shape = (batch, text_length, hidden_dim)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, video_length, text_length)
  embeddings = PositionalEncoding(hidden_dim)(inputs);
  outputs = tf.keras.layers.Dropout(rate = kwargs['drop_rate'])(embeddings);
  for i in range(num_layers):
    outputs = DecoderLayer(hidden_dim, num_heads, drop_rate = kwargs['drop_rate'], origin_shape = kwargs['origin_shape'])([outputs, code, mask]);
  return tf.keras.Model(inputs = (inputs, code, mask), outputs = outputs);

def Transformer(encoder_layers = 2, decoder_layers = 2, hidden_dim = 128, num_heads = 16, origin_shape = (64, 64), text_vocab_size = None, video_vocab_size = 10000, **kwargs):
  text_inputs = tf.keras.Input((None,)); # inputs.shape = (batch, text_length)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length, text_length)
  # INFO: to avoid repeat calculating embedding of leading frames, the input uses code from VQVAE, but leading frames
  # NOTE: video_top_inputs.shape[1] = origin_shape[1] // 8 * origin_shape[2] // 8 * frame_number
  # NOTE: video_bottom_inputs.shape[1] = origin_shape[1] // 4 * origin_shape[2] // 4 * frame_number
  video_inputs = tf.keras.Input((None,)); # video_top_inputs.shape = (batch, frame * 64 * 64 + 2,)
  
  text_embed = tf.keras.layers.Embedding(text_vocab_size, hidden_dim)(text_inputs);
  text_embed = tf.keras.layers.Lambda(lambda x, d: tf.math.sqrt(tf.cast(d, dtype = tf.float32)) * x, arguments = {'d': hidden_dim})(text_embed);
  video_embed = tf.keras.layers.Embedding(video_vocab_size, hidden_dim)(video_inputs);
  video_embed = tf.keras.layers.Lambda(lambda x, d: tf.math.sqrt(tf.cast(d, dtype = tf.float32)) * x, arguments = {'d': hidden_dim})(video_embed);
  
  text_code = TransEncoder(encoder_layers, hidden_dim, num_heads, drop_rate = kwargs['drop_rate'])([text_embed, mask]); # text_code.shape = (batch, text_length, hidden_dim)
  video_code = TransDecoder(decoder_layers, hidden_dim, num_heads, drop_rate = kwargs['drop_rate'], origin_shape = (origin_shape[0], origin_shape[1]))([video_embed, text_code, mask]);
  video_pred = tf.keras.layers.Dense(units = video_vocab_size, activation = tf.keras.activations.softmax)(video_code);
  return tf.keras.Model(inputs = (text_inputs, mask, video_inputs), outputs = video_pred);

class GODIVA(tf.keras.Model):
  def __init__(self, img_size = 64, video_length = 16, text_vocab_size = None, video_vocab_size = 10000, **kwargs):
    super(GODIVA, self).__init__(**kwargs);
    self.video_length = video_length;
    self.transformer = Transformer(origin_shape = (img_size // 4, img_size // 4), text_vocab_size = text_vocab_size + 2, video_vocab_size = video_vocab_size + 2, drop_rate = 0.2);
    self.VIDEO_SOS = video_vocab_size;
    self.VIDEO_EOS = video_vocab_size + 1;
    self.frame_token_num = img_size // 4 * img_size // 4;
  def call(self, inputs):
    # inputs.shape = (batch, length)
    text = inputs[0]; # text.shape = (batch, text_length)
    mask = inputs[1]; # mask.shape = (batch, 1, 1, text_length)
    total_tokens = tf.ones((tf.shape(text)[0], self.frame_token_num), dtype = tf.int64) * self.VIDEO_SOS; # tokens.shape = (batch, (origin_shape // 4) ** 2)
    total_preds = tf.ones((tf.shape(text)[0], 0, self.transformer.output[0].shape[-1]), dtype = tf.float32); # preds.shape = (batch, 0, video_vocab_size + 2)
    for i in range(self.video_length + 1):
      pred = self.transformer([text, mask, total_tokens]); # bottom_pred.shape = (batch, length, video_vocab_size + 2)
      tokens = tf.math.argmax(pred, axis = -1); # tokens.shape = (batch, length)
      total_tokens = tf.concat([total_tokens, tokens[:,-self.frame_token_num:]], axis = 1); # bottom_tokens.shape = (batch, length + bottom_frame_token_num)
      total_preds = tf.concat([total_preds, pred[:,-self.frame_token_num:,:]], axis = 1); # bottom_tokens.shape = (batch, length + bottom_frame_token_num, video_vocab_size + 2)
    # NOTE: total_preds.shape = (batch, (video_length + 1) * frame_token_num, video_vocab_size + 2)
    return total_preds;

if __name__ == "__main__":

  tokens = np.random.randint(low = 0, high = 10, size = (1, 34));
  mask = np.random.randint(low = 0, high = 2, size = (1, 1, 1, 34));
  godiva = GODIVA(text_vocab_size = 10);
  preds = godiva([tokens, mask]);
  print(preds.shape);
  godiva.save_weights('godiva.h5');
