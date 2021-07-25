#!/usr/bin/python3

from os.path import exists, join;
from absl import flags, app;
import tensorflow as tf;
from models import VQVAE_Trainer, Quantize, QuantizeEma;
from create_dataset import parse_function_generator, load_dataset;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', default = 128, help = 'batch size');
flags.DEFINE_integer('img_size', default = 64, help = 'image size');
flags.DEFINE_enum('type', default = 'ema_update', enum_values = ['original', 'ema_update'], help = 'quantization type');
flags.DEFINE_string('train_dir', default = 'trainsets', help = 'directory containing training samples');
flags.DEFINE_string('test_dir', default = 'testsets', help = 'directory containing testing samples');
flags.DEFINE_boolean('save_model', default = False, help = 'whether to save model');

def recon_loss(labels, outputs):
  return tf.keras.losses.MeanSquaredError()(labels, outputs);

def quant_loss(_, outputs):
  return outputs;

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, trainer, eval_freq = 100):
    self.trainer = trainer;
    self.eval_freq = eval_freq;
    testset = load_dataset(FLAGS.test_dir).map(parse_function_generator(output_size=(FLAGS.img_size, FLAGS.img_size))).repeat(-1).batch(1);
    self.iter = iter(testset);
    self.recon_loss = tf.keras.metrics.Mean(name = 'recon_loss', dtype = tf.float32);
    self.quant_loss = tf.keras.metrics.Mean(name = 'quant_loss', dtype = tf.float32);
    self.log = tf.summary.create_file_writer('./checkpoints');
  def on_batch_begin(self, batch, logs = None):
    pass;
  def on_batch_end(self, batch, logs = None):
    image, label_dict = next(self.iter);
    recon, diff = self.trainer(image); # recon.shape = (batch, 256, 256, 3)
    recon_loss = tf.keras.losses.MeanSquaredError()(image, recon);
    self.recon_loss.update_state(recon_loss);
    self.quant_loss.update_state(diff);
    if batch % self.eval_freq == 0:
      recon = recon * tf.reshape([0.5,0.5,0.5], (1, 1, 1, -1)) + tf.reshape([0.5,0.5,0.5], (1, 1, 1, -1));
      recon = tf.cast(recon * 255., dtype = tf.uint8);
      with self.log.as_default():
        tf.summary.scalar('reconstruction loss', self.recon_loss.result(), step = self.trainer.optimizer.iterations);
        tf.summary.scalar('quantize loss', self.quant_loss.result(), step = self.trainer.optimizer.iterations);
        tf.summary.image('reconstructed image', recon, step = self.trainer.optimizer.iterations);
      self.recon_loss.reset_states();
      self.quant_loss.reset_states();
  def on_epoch_begin(self, epoch, logs = None):
    pass;
  def on_epoch_end(self, batch, logs = None):
    pass;

def pretrain():
  if exists('./checkpoints/ckpt'):
    trainer = tf.keras.models.load_model('./checkpoints/ckpt', custom_objects = {'tf': tf, 'Quantize': Quantize, 'QuantizeEma': QuantizeEma, 'recon_loss': recon_loss, 'quant_loss': quant_loss}, compile = True);
    optimizer = trainer.optimizer;
  else:
    trainer = VQVAE_Trainer(quantize_type = FLAGS.type);
    optimizer = tf.keras.optimizers.Adam(3e-4);
    trainer.compile(optimizer = optimizer, loss = {'output_1': recon_loss, 'output_2': quant_loss}, loss_weights = {'output_1': 1,'output_2': 1});

  # load imagenet dataset
  trainset = load_dataset(FLAGS.train_dir).map(parse_function_generator(output_size=(FLAGS.img_size, FLAGS.img_size))).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = load_dataset(FLAGS.test_dir).map(parse_function_generator(output_size=(FLAGS.img_size, FLAGS.img_size))).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 10000),
    SummaryCallback(trainer)
  ];
  trainer.fit(trainset, epochs = 560, validation_data = testset, callbacks = callbacks);
  trainer.save(join('models', 'trainer.h5'));

def save_model():
  trainer = VQVAE_Trainer(quantize_type = FLAGS.type);
  trainer.load_weights('./checkpoints/ckpt');
  if FLAGS.type == 'ema_update':
    trainer.layers[1].layers[4].set_trainable(False);
    trainer.layers[1].layers[8].set_trainable(False);
  trainer.layers[1].save(join('models', 'encoder.h5'));
  trainer.layers[1].save(join('models', 'decoder.h5'));

def main(unused_argv):
  if FLAGS.save_model:
    save_model();
  else:
    pretrain();

if __name__ == "__main__":
  app.run(main);
