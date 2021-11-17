#!/usr/bin/python3

import h5py;
from os.path import join;
import numpy as np;
import tensorflow as tf;
from vqvae import Quantize, QuantizeEma;

class SampleGenerator(object):  
  def __init__(self, filename):
    f = h5py.File(filename, 'r');
    self.data_train = np.array(f['mnist_gif_train']);
    self.captions_train = np.array(f['mnist_captions_train']);
    self.data_val = np.array(f['mnist_gif_val']);
    self.captions_val = np.array(f['mnist_captions_val']);
    f.close();
  def sample_generator(self, is_trainset = True):
    data = self.data_train if is_trainset else self.data_val;
    caption = self.captions_train if is_trainset else self.captions_val;
    def gen():
      for i in range(data.shape[0]):
        sample = np.transpose(data[i], (0,2,3,1));
        text = caption[i];
        yield sample, text;
    return gen;
  def get_trainset(self,):
    return tf.data.Dataset.from_generator(self.sample_generator(True), (tf.float32, tf.int64), (tf.TensorShape([16,64,64,1]), tf.TensorShape([9,])));
  def get_testset(self):
    return tf.data.Dataset.from_generator(self.sample_generator(False), (tf.float32, tf.int64), (tf.TensorShape([16,64,64,1]), tf.TensorShape([9,])));

class parse_function(object):
  def __init__(self, dataset = 'single', img_size = 64, encoder = join('models', 'encoder.h5'), decoder = join('models', 'decoder.h5')):
    assert dataset in ['single', 'double'];
    self.encoder = tf.keras.models.load_model(encoder, custom_objects = {'tf': tf, 'Quantize': Quantize, 'QuantizeEma': QuantizeEma});
    self.decoder = tf.keras.models.load_model(decoder, custom_objects = {'tf': tf});
    quantize = self.encoder.get_layer('top_quantize');
    embed_tab = quantize.get_embed();
    vocab_size = tf.shape(embed_tab)[-1];
    if dataset == 'single':
      from dataset.mnist_caption_single import dictionary;
      text_vocab_size = len(dictionary);
    elif dataset == 'double':
      from dataset.mnist_caption_two_digit import dictionary;
      text_vocab_size = len(dictionary);
    self.TEXT_SOS = tf.cast(text_vocab_size, dtype = tf.int64);
    self.TEXT_EOS = tf.cast(text_vocab_size + 1, dtype = tf.int64);
    self.VIDEO_SOS = tf.cast(vocab_size, dtype = tf.int64);
    self.VIDEO_EOS = tf.cast(vocab_size + 1, dtype = tf.int64);
    self.frame_token_num = img_size // 4 * img_size // 4;
  def parse_function(self, sample, text):
    # sample.shape = (length, height, width, channel)
    # text.shape = (length)
    # 1) text preprocess
    padded_text = tf.concat([[self.TEXT_SOS], text, [self.TEXT_EOS]], axis = 0); # text.shape = (length + 2)
    # NOTE: mask padded tokens, as all sample in this datasets has the same length, no padded token is used
    mask = tf.concat([tf.cast([1,], dtype = tf.int64), tf.ones_like(text), tf.cast([1,], dtype = tf.int64)], axis = 0); # mask.shape (length + 2)
    mask = tf.reshape(mask, (1,1,-1)); # mask.shape = (1, 1, length + 2)
    # 2) video preprocess
    sample = (sample / 255. - 0.5) / 0.5;
    # tile to 3-channel images
    sample = tf.tile(sample, (1,1,1,3)); # samples.shape = (lenght, h, w, 3)
    # tokens.shape = (length, h/4, w/4)
    _, tokens, _ = self.encoder(sample);
    tokens = tf.reshape(tokens, (-1,)); # video_token_b.shape = (length * h/4 * w/4)
    outputs = tf.concat([tokens, tf.ones((self.frame_token_num,), dtype = tf.int64) * self.VIDEO_EOS], axis = 0); # inputs.b.shape = ((length + 2) * h/4 * w/4)
    return (padded_text, mask), outputs;

if __name__ == "__main__":

  import cv2;
  generator = SampleGenerator('mnist_single_gif.h5');
  parse_func = parse_function();
  trainset = generator.get_trainset().map(parse_func.parse_function).batch(1);
  decoder = parse_func.decoder;
  embed_tab = tf.transpose(parse_func.encoder.get_layer('top_quantize').get_embed()); # embed_tab.shape = (n_embed, embed_dim)
  cv2.namedWindow('sample');
  for (padded_text, mask), outputs in trainset:
    tokens = outputs[:,:-parse_func.frame_token_num]; # tokens.shape = (1, video_length * frame_token_num)
    tokens = tf.reshape(tokens, (1, 16, 16, 16)); # tokens.shape = (1, video_length = 16, height = 16, width = 16)
    embeds = tf.gather(embed_tab, tokens); # embeds.shape = (1, video_length, height, width, embed_dim)
    frames = list();
    for i in range(16):
      frame_embeds = embeds[:,i,...]; # frame_embeds.shape = (1, height, width, embed_dim)
      recon = decoder(frame_embeds); # recon.shape = (1, 64, 64, 3)
      recon = tf.cast((recon * 0.5 + 0.5) * 255., dtype = tf.uint8).numpy()[0]; # recon.shape = (64,64,3)
      frames.append(recon);
    for image in frames:
      cv2.imshow('sample',image.astype(np.uint8));
      cv2.waitKey(50);
