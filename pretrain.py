#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from absl import flags, app;
import tensorflow as tf;
from vqvae import VQVAE_Trainer, Quantize, QuantizeEma;
from create_dataset import parse_function_generator, load_dataset;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', default = 128, help = 'batch size');
flags.DEFINE_integer('img_size', default = 64, help = 'image size');
flags.DEFINE_integer('token_num', default = 10000, help = 'how many tokens in code book');
flags.DEFINE_enum('type', default = 'ema_update', enum_values = ['original', 'ema_update'], help = 'quantization type');
flags.DEFINE_string('train_dir', default = 'trainsets', help = 'directory containing training samples');
flags.DEFINE_string('test_dir', default = 'testsets', help = 'directory containing testing samples');
flags.DEFINE_enum('mode', default = 'train', enum_values = ['train', 'save', 'test'], help = 'mode to run');
flags.DEFINE_string('img', default = None, help = 'image to predict on');

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
    trainer = VQVAE_Trainer(n_embed = FLAGS.token_num, quantize_type = FLAGS.type);
    optimizer = tf.keras.optimizers.Adam(3e-4);
    trainer.compile(optimizer = optimizer, loss = {'decoder': recon_loss, 'encoder': quant_loss}, loss_weights = {'decoder': 1,'encoder': 1});

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
    trainer.get_layer('encoder').get_layer('top_quantize').set_trainable(False);
  if not exists('models'): mkdir('models');
  trainer.get_layer('encoder').save(join('models', 'encoder.h5'));
  trainer.get_layer('decoder').save(join('models', 'decoder.h5'));

def test():
  import cv2;
  img = cv2.imread(FLAGS.img);
  if img is None:
    raise Exception('invalid image path');
  encoder = tf.keras.models.load_model(join('models', 'encoder.h5'), custom_objects = {'Quantize': Quantize, 'QuantizeEma': QuantizeEma});
  decoder = tf.keras.models.load_model(join('models', 'decoder.h5'));
  resized = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size)) / 255.;
  normalized = (resized - tf.reshape([0.5,0.5,0.5], (1,1,-1))) / tf.reshape([0.5,0.5,0.5], (1,1,-1));
  inputs = np.expand_dims(normalized, axis = 0);
  quantized_t, cluster_index_t, loss_t, quantized_b, cluster_index_b, loss_b = encoder(inputs);
  recon = decoder([quantized_t, quantized_b]);
  recon_img = tf.squeeze(recon, axis = 0).numpy() * np.reshape([0.5,0.5,0.5], (1, 1, -1)) + np.reshape([0.5,0.5,0.5], (1, 1, -1));
  recon_img = 255. * recon_img;
  cv2.imshow('reconstructed', recon_img.astype(np.uint8));
  cv2.waitKey();

def main(unused_argv):
  if FLAGS.mode == 'save':
    save_model();
  elif FLAGS.mode == 'train':
    pretrain();
  elif FLAGS.mode == 'test':
    test();
  else:
    raise Exception('unknow mode!');

if __name__ == "__main__":
  app.run(main);
