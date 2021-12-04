#!/usr/bin/python3

from math import sqrt;
from os import mkdir;
from os.path import exists, join;
from absl import app, flags;
import tensorflow as tf;
from godiva import GODIVA;
from vqvae import Quantize, QuantizeEma;
from dataset.sample_generator import SampleGenerator, parse_function;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', default = 1, help = 'batch size');
flags.DEFINE_enum('dataset', default = 'single', enum_values = ['single', 'double'], help = 'which dataset to train on');
flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'path to checkpoint');

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, godiva, validationset, eval_freq = 100):
    self.godiva = godiva;
    self.decoder = tf.keras.models.load_model(join('models', 'decoder.h5'));
    encoder = tf.keras.models.load_model(join('models', 'encoder.h5'), custom_objects = {'tf': tf, 'Quantize': Quantize, 'QuantizeEma': QuantizeEma});
    self.embed_tab = tf.transpose(encoder.get_layer('top_quantize').get_embed()); # embed_tab.shape = (n_embed, embed_dim)
    self.eval_freq = eval_freq;
    self.iter = iter(validationset);
    self.log = tf.summary.create_file_writer(FLAGS.checkpoint);
  def on_batch_end(self, batch, logs = None):
    if batch % self.eval_freq == 0:
      (padded_text, mask), label = next(self.iter);
      preds = self.godiva([padded_text, mask]);
      preds = preds[...,:-1]; # preds.shape = (batch, video_length * frame_token_num, video_vocab_size)
      tokens = tf.math.argmax(preds, axis = -1); # tokens.shape = (batch, video_length * frame_token_num)
      embeds = tf.gather(self.embed_tab, tokens); # embeds.shape = (batch, video_length * frame_token_num, embed_dim)
      embeds = tf.reshape(embeds, (-1, embeds.shape[1] // self.godiva.frame_token_num, int(sqrt(self.godiva.frame_token_num)), int(sqrt(self.godiva.frame_token_num)), embeds.shape[-1])); # embed.shape = (batch, video_length, h // 4, w // 4, embed_dim)
      video = list();
      for i in range(embeds.shape[1]):
        frame = embeds[:, i, ...]; # frame.shape = (batch, h//4, w//4, embed_dim)
        recon = self.decoder(frame); # recon.shape = (batch, h, w, 3)
        video.append(recon);
      video = tf.stack(video, axis = 1); # video.shape = (batch, video_length, h, w, 3);
      assert video.shape[1] == 16;
      video = tf.transpose(tf.reshape(video, (video.shape[0], 4, 4, video.shape[2], video.shape[3], video.shape[4])), (0,1,3,2,4,5));
      video = tf.reshape(video, (video.shape[0], video.shape[1] * video.shape[2], video.shape[3] * video.shape[4], video.shape[5])); # video.shape = (batch, 4 * h, 4 * w, 3)
      video = tf.cast((video * 0.5 + 0.5) * 255., dtype = tf.uint8);
      with self.log.as_default():
        for key, value in logs.items():
          tf.summary.scalar(key, value, step = self.godiva.optimizer.iterations);
        tf.summary.image('generated', video, step = self.godiva.optimizer.iterations);
        tf.summary.scalar('lr', self.godiva.optimizer._decayed_lr(tf.float32), step = self.godiva.optimizer.iterations);

def main(unused_argv):

  if FLAGS.dataset == 'single':
    from dataset.mnist_caption_single import dictionary;
    text_vocab_size = len(dictionary);
    filename = 'mnist_single_gif.h5';
  elif argv[1] == 'double':
    from dataset.mnist_caption_two_digit import dictionary;
    text_vocab_size = len(dictionary);
    filename = 'mnist_two_gif.h5';

  if exists(join(FLAGS.checkpoint, 'ckpt')):
    godiva = tf.keras.models.load_model(join(FLAGS.checkpoint, 'ckpt'), custom_objects = {'tf': tf, 'Quantize': Quantize, 'QuantizeEma': QuantizeEma}, compile = True);
    optimizer = godiva.optimizer;
  else:
    godiva = GODIVA(text_vocab_size = text_vocab_size);
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(3e-3, decay_steps = 40000, decay_rate = 0.8, staircase = True));
    godiva.compile(optimizer = optimizer,
                   loss = [tf.keras.losses.SparseCategoricalCrossentropy(name = 'ce_loss')],
                   metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name = 'acc')]);
  # generate dataset
  dataset_generator = SampleGenerator(filename);
  parse_func = parse_function();
  trainset = dataset_generator.get_trainset().map(parse_func.parse_function).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = dataset_generator.get_testset().map(parse_func.parse_function).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  validationset = dataset_generator.get_testset().map(parse_func.parse_function).batch(1);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.checkpoint),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.checkpoint, 'ckpt'), save_freq = 1000),
    SummaryCallback(godiva, validationset, eval_freq = 100),
  ];
  godiva.fit(trainset, epochs = 560, validation_data = testset, callbacks = callbacks);
  godiva.save_weights(join('models', 'godiva_weights.h5'));

if __name__ == "__main__":

  app.run(main);
