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
    top_quantize = self.encoder.layers[4];
    top_embed_tab = top_quantize.get_embed();
    top_vocab_size = tf.shape(top_embed_tab)[-1];
    bottom_quantize = self.encoder.layers[8];
    bottom_embed_tab = bottom_quantize.get_embed();
    bottom_vocab_size = tf.shape(bottom_embed_tab)[-1];
    if dataset == 'single':
      from dataset.mnist_caption_single import dictionary;
      text_vocab_size = len(dictionary);
    elif dataset == 'double':
      from dataset.mnist_caption_two_digit import dictionary;
      text_vocab_size = len(dictionary);
    self.TEXT_SOS = tf.cast(text_vocab_size, dtype = tf.int64);
    self.TEXT_EOS = tf.cast(text_vocab_size + 1, dtype = tf.int64);
    self.top_SOS = tf.cast(top_vocab_size, dtype = tf.int64);
    self.top_EOS = tf.cast(top_vocab_size + 1, dtype = tf.int64);
    self.bottom_SOS = tf.cast(bottom_vocab_size, dtype = tf.int64);
    self.bottom_EOS = tf.cast(bottom_vocab_size + 1, dtype = tf.int64);
    self.top_frame_token_num = img_size // 8 * img_size // 8;
    self.bottom_frame_token_num = img_size // 4 * img_size // 4;
  def parse_function(self, sample, text):
    # sample.shape = (length, height, width, channel)
    # text.shape = (length)
    # 1) text preprocess
    text = tf.concat([self.TEXT_SOS, text, self.TEXT_EOS], axis = 0); # text.shape = (length + 2)
    mask = tf.concat([tf.cast(0, dtype = tf.int64), tf.ones_like(text), tf.cast(0, dtype = tf.int64)], axis = 0); # mask.shape (length + 2)
    mask = tf.reshape(mask, (1,1,-1)); # mask.shape = (1, 1, length + 2)
    # 2) video preprocess
    sample = sample / 255. - 0.5;
    # tile to 3-channel images
    sample = tf.tile(sample, (1,1,1,3)); # samples.shape = (lenght, h, w, 3)
    # tokens_t.shape = (length, h/8, h/8), tokens_b.shape = (length, h/4, w/4)
    _, tokens_t, _, _, tokens_b, _ = self.encoder(sample);
    tokens_t = tf.reshape(tokens_t, (-1,)); # video_token_t.shape = (length * h/8 * w/8)
    tokens_b = tf.reshape(tokens_b, (-1,)); # video_token_b.shape = (length * h/4 * w/4)
    outputs_t = tf.concat([tokens_t, tf.ones((self.top_frame_token_num,), dtype = tf.int64) * self.top_EOS], axis = 0); # inputs_t.shape = ((length + 2) * h/8 * w/8)
    outputs_b = tf.concat([tokens_b, tf.ones((self.bottom_frame_token_num,), dtype = tf.int64) * self.bottom_EOS], axis = 0); # inputs.b.shape = ((length + 2) * h/4 * w/4)
    return (text, mask), (outputs_t, outputs_b);

if __name__ == "__main__":

  import cv2;
  generator = SampleGenerator('mnist_single_gif.h5');
  trainset = generator.get_trainset();
  testset = generator.get_testset();
  cv2.namedWindow('sample');
  for sample, caption in trainset:
    print(caption);
    for image in sample:
      cv2.imshow('sample',image.numpy().astype(np.uint8));
      cv2.waitKey(50);
  generator = SampleGenerator('mnist_two_gif.h5');
  trainset = generator.get_trainset();
  testset = generator.get_testset();
