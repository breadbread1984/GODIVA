#!/usr/bin/python3

from os.path import exists;
import tensorflow as tf;
from models import VQVAE_Trainer;
from create_dataset import parse_function_generator, load_dataset;

batch_size = 128;

def main(train_dir, test_dir):

  trainer = VQVAE_Trainer();
  if exists('./checkpoints/chkpt'): trainer.load_weights('./checkpoints/ckpt/variables/variables');
  optimizer = tf.keras.optimizers.Adam(3e-4);
  trainer.compile(optimizer = optimizer,
                  loss = {'decoder': lambda labels, outputs: tf.keras.losses.MeanSquaredError(labels, outputs),
                          'diff': lambda dummy, outputs: outputs},
                  loss_weights = {'decoder': 1,'diff': 0.25});
  class SummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, eval_freq = 100):
      self.eval_freq = eval_freq;
      testset = load_dataset(test_dir).map(parse_function_generator()).repeat(-1).batch(1);
      self.iter = iter(testset);
      self.recon_loss = tf.keras.metrics.Mean(name = 'recon_loss', dtype = tf.float32);
      self.quant_loss = tf.keras.metrics.Mean(name = 'quant_loss', dtype = tf.float32);
      self.log = tf.summary.create_file_writer('./checkpoints');
    def on_batch_begin(self, batch, logs = None):
      pass;
    def on_batch_end(self, batch, logs = None):
      image, (image, label) = next(self.iter);
      recon, diff = trainer(image); # recon.shape = (batch, 256, 256, 3)
      recon_loss = tf.keras.losses.MeanSquaredError(image, recon);
      self.recon_loss.update_state(recon_loss);
      self.quant_loss.update_state(diff);
      if batch % self.eval_freq == 0:
        recon = recon * 128. + tf.reshape([123.68, 116.78, 103.94], (1, 1, 1, -1));
        with self.log.as_default():
          tf.summary.scalar('reconstruction loss', self.recon_loss.result(), step = optimizer.iterations);
          tf.summary.scalar('quantize loss', self.quant_loss.result(), step = optimizer.iterations);
          tf.summary.image('reconstructed image', recon, step = optimizer.iterations);
        self.recon_loss.reset_states();
        self.quant_loss.reset_states();
    def on_epoch_begin(self, epoch, logs = None):
      pass;
    def on_epoch_end(self, batch, logs = None):
      pass;
  
  # load imagenet dataset
  trainset = load_dataset(train_dir).map(parse_function_generator()).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = load_dataset(test_dir).map(parse_function_generator()).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.Tensorboard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 10000),
    SummaryCallback()
  ];
  trainer.fit(trainset, epochs = 560, validation_data = testset, callbacks = callbacks);
  trainer.encoder.save('encoder.h5');
  trainer.decoder.save('decoder.h5');

if __name__ == "__main__":
  from sys import argv;
  if len(argv) != 3:
    print('Usage: %s <train_dir> <test_dir>' % argv[0]);
    exit(1);
  main(argv[1], argv[2]);
