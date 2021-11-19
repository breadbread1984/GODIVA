#!/usr/bin/python3

from os.path import join;
from absl import flags, app;
import numpy as np;
import cv2;
import tensorflow as tf;
from vqvae import *;
from dataset.sample_generator import SampleGenerator, parse_function;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_enum('dataset', default = 'single', enum_values = ['single', 'double'], help = 'which dataset to train on');
  flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'path to checkpoint');

def main(unused_argv):
  encoder = tf.keras.models.load_model(join('models', 'encoder.h5'), custom_objects = {'tf': tf, 'Quantize': Quantize, 'QuantizeEma': QuantizeEma});
  decoder = tf.keras.models.load_model(join('models', 'decoder.h5'));
  embed_tab = tf.transpose(encoder.get_layer('top_quantize').get_embed()); # embed_tab.shape = (n_embed, embed_dim)
  godiva = tf.keras.models.load_model(join(FLAGS.checkpoint, 'ckpt'), custom_objects = {'tf': tf, 'Quantize': Quantize, 'QuantizeEma': QuantizeEma}, compile = True);
  if FLAGS.dataset == 'single':
    from dataset.mnist_caption_single import dictionary;
    text_vocab_size = len(dictionary);
    filename = 'mnist_single_gif.h5';
  elif argv[1] == 'double':
    from dataset.mnist_caption_two_digit import dictionary;
    text_vocab_size = len(dictionary);
    filename = 'mnist_two_gif.h5';
  dataset_generator = SampleGenerator(filename);
  parse_func = parse_function();
  testset = dataset_generator.get_testset().map(parse_func.parse_function).batch(1);
  for (padded_text, mask), label in testset:
    preds = godiva([padded_text, mask]);
    preds = preds[...,:-1]; # preds.shape = (batch, video_length * frame_token_num, video_vocab_size)
    tokens = tf.math.argmax(preds, axis = -1); # tokens.shape = (batch, video_length * frame_token_num)
    embeds = tf.gather(embed_tab, tokens); # embeds.shape = (batch, video_length * frame_token_num, embed_dim)
    embeds = tf.reshape(embeds, (-1, embeds.shape[1] // godiva.frame_token_num, int(sqrt(godiva.frame_token_num)), int(sqrt(godiva.frame_token_num)), embeds.shape[-1])); # embed.shape = (batch, video_length, h // 4, w // 4, embed_dim)
    video = list();
    for i in range(embeds.shape[1]):
      frame = embeds[:, i, ...]; # frame.shape = (batch, h//4, w//4, embed_dim)
      recon = decoder(frame); # recon.shape = (batch, h, w, 3)
      video.append(recon);
    video = tf.stack(video, axis = 1); # video.shape = (batch, video_length, h, w, 3);
    video = tf.cast(video[0], dtype = tf.uint8).numpy(); # video.shape = (video_length, h, w, 3);
    for frame in video:
      cv2.imshow('frame', frame);
      cv2.waitKey(25);

if __name__ == "__main__":
  add_options();
  app.run(main);

