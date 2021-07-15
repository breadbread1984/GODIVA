#!/usr/bin/python3

import tensorflow as tf;
from models import GODIVA;
from dataset.sample_generator import SampleGenerator, parse_function;

batch_size = 4;

def main(filename, text_vocab_size):
  # generate dataset
  dataset_generator = SampleGenerator(filename);
  trainset = dataset_generator.get_trainset().map(parse_function).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  trainset_iter = iter(trainset);
  # create godiva
  godiva = GODIVA(text_vocab_size = text_vocab_size);
  while True:
    # video.shape = (batch, frame, h, w, 3)
    # text.shape = (batch, seq_length)
    video, text = next(trainset_iter);
    video = tf.reshape(video, (tf.shape(video)[0] * tf.shape(video)[1], *tf.shape(video)[-3:])); # video.shape = (batch * length, h, w, 1)
    video = tf.tile(video, (1,1,1,3)); # video.shape = (batch * length, h, w, 3)
    # video_token_t.shape = (batch * length, h/8, w/8,)
    # video_token_b.shape = (batch * length, h/4, w/4,)
    quantized_t, video_token_t, _, quantized_b, video_token_b, _ = godiva.encoder(video);
    video_token_t = tf.reshape(video_token_t, (batch_size, 16, -1)); # video_token_t.shape = (batch, length, h/8 * w/8)
    video_token_b = tf.reshape(video_token_b, (batch_size, 16, -1)); # video_token_b.shape = (batch, length, h/4 * w/4)

  # TODO
  
if __name__ == "__main__":

  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s (single|double)' % argv[0]);ls
    dfs
    exit(1);
  assert argv[1] in ['single','double'];
  if argv[1] == 'single':
    from dataset.mnist_caption_single import dictionary;
    text_vocab_size = len(dictionary);
  elif argv[1] == 'double':
    from dataset.mnist_caption_two_digit import dictionary;
    text_vocab_size = len(dictionary);
  main('mnist_single_gif.h5', text_vocab_size);
