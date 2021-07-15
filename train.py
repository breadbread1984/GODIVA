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
  # TODO
  
if __name__ == "__main__":

  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s (single|double)' % argv[0]);
    exit(1);
  assert argv[1] in ['single','double'];
  if argv[1] == 'single':
    from dataset.mnist_caption_single import dictionary;
    text_vocab_size = len(dictionary);
  elif argv[1] == 'double':
    from dataset.mnist_caption_two_digit import dictionary;
    text_vocab_size = len(dictionary);
  main('mnist_single_gif.h5', text_vocab_size);
