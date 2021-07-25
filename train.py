#!/usr/bin/python3

from os import mkdir;
from os.path import exists;
from absl import app, flags;
import tensorflow as tf;
from models import GODIVA, Quantize, QuantizeEma;
from dataset.sample_generator import SampleGenerator, parse_function;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', default = 4, help = 'batch size');
flags.DEFINE_enum('dataset', default = 'single', enum_values = ['single', 'double'], help = 'which dataset to train on');

def main(unused_argv):

  if FLAGS.dataset == 'single':
    from dataset.mnist_caption_single import dictionary;
    text_vocab_size = len(dictionary);
    filename = 'mnist_single_gif.h5';
  elif argv[1] == 'double':
    from dataset.mnist_caption_two_digit import dictionary;
    text_vocab_size = len(dictionary);
    filename = 'mnist_two_gif.h5';

  if exists('./checkpoints/ckpt'):
    godiva = tf.keras.models.load_model('./checkpoints/ckpt', custom_objects = {'tf': tf, 'Quantize': Quantize, 'QuantizeEma': QuantizeEma}, compile = True);
    optimizer = godiva.optimizer;
  else:
    godiva = GODIVA(text_vocab_size = text_vocab_size);
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(3e-4, decay_steps = 20000, decay_rate = 0.97));
    godiva.compile(optimizer = optimizer,
                   loss = {tf.keras.losses.SparseCategoricalCrossentropy(), tf.keras.losses.SparseCategoricalCrossentropy()},
                   metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseCategoricalAccuracy()]);
  # generate dataset
  dataset_generator = SampleGenerator(filename);
  parse_func = parse_function();
  trainset = dataset_generator.get_trainset().map(parse_func.parse_function).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = dataset_generator.get_testset().map(parse_func.parse_function).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 1000)
  ];
  godiva.fit(trainset, epochs = 560, validation_data = testset, callbacks = callbacks);
  godiva.save_weights(join('models', 'godiva_weights.h5'));

if __name__ == "__main__":

  app.run(main);
