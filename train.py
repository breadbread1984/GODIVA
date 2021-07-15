#!/usr/bin/python3

from os import mkdir;
from os.path import exists;
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
  # optimizer
  top_optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 20000, decay_rate = 0.97));
  bottom_optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 20000, decay_rate = 0.97));
  # checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(godiva = godiva, top_optimizer = top_optimizer, bottom_optimizer = bottom_optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # summary
  log = tf.summary.create_file_writer('checkpoints');
  top_avg_loss = tf.keras.metrics.Mean(name = 'top loss', dtype = tf.float32);
  bottom_avg_loss = tf.keras.metrics.Mean(name = 'bottom loss', dtype = tf.float32);
  while True:
    # 1) load sample
    video, text = next(trainset_iter);
    video = tf.reshape(video, (tf.shape(video)[0] * tf.shape(video)[1], *tf.shape(video)[-3:])); # video.shape = (batch * length, h, w, 1)
    video = tf.tile(video, (1,1,1,3)); # video.shape = (batch * length, h, w, 3)
    # 2) generate transformer's inputs and outputs(ground truth)
    quantized_t, video_token_t, _, quantized_b, video_token_b, _ = godiva.encoder(video);
    video_token_t = tf.reshape(video_token_t, (batch_size, -1)); # video_token_t.shape = (batch, length * h/8 * w/8)
    video_token_b = tf.reshape(video_token_b, (batch_size, -1)); # video_token_b.shape = (batch, length * h/4 * w/4)
    top_sos = tf.ones((batch_size, godiva.top_frame_token_num), dtype = tf.int64) * godiva.SOS;
    bottom_sos = tf.ones((batch_size, godiva.bottom_frame_token_num), dtype = tf.int64) * godiva.SOS;
    top_eos = tf.ones((batch_size, godiva.top_frame_token_num), dtype = tf.int64) * godiva.EOS;
    bottom_eos = tf.ones((batch_size, godiva.bottom_frame_token_num), dtype = tf.int64) * godiva.EOS;
    top_inputs = tf.concat([top_sos, video_token_t], axis = -1);
    bottom_inputs = tf.concat([bottom_sos, video_token_b], axis = -1);
    top_outputs = tf.concat([video_token_t, top_eos], axis = -1);
    bottom_outputs = tf.concat([video_token_b, bottom_eos], axis = -1);
    # 3) predict and loss
    with tf.GradientTape() as top_tape:
      top_preds = godiva.top_transformer([text, top_inputs]); # top_preds.shape = (batch, length, video_vocab_size + 2)
      top_loss = tf.keras.losses.SparseCategoricalCrossentropy(top_outputs, top_preds)
    with tf.GradientTape() as bottom_tape:
      bottom_preds = godiva.bottom_transformer([text, bottom_inputs]); # bottom_preds.shape = (batch, length, video_vocab_size + 2)
      bottom_loss = tf.keras.losses.SparseCategoricalCrossentropy(bottom_outputs, bottom_preds);
    # 4) gradient
    top_avg_loss.update_state(top_loss);
    bottom_avg_loss.update_state(bottom_loss);
    top_grads = top_tape.gradient(top_loss, godiva.top_transformer.trainable_variables);
    bottom_grads = bottom_tape.gradient(bottom_loss, godiva.bottom_transformer.trainable_variables);
    top_optimizer.apply_gradient(zip(top_grads, top_transformer.trainable_variables));
    bottom_optimizer.apply_gradient(zip(bottom_grads, bottom_transformer.trainable_variables));
    # 5) save checkpoint
    if tf.equal(top_optimizer.iterations % 10000, 0):
      checkpoint.save(join('checkpoints', 'ckpt'));
    if tf.equal(top_optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scale('top loss', top_avg_loss.result(), step = top_optimizer.iterations);
        tf.summary.scale('bottom loss', bottom_avg_loss.result(), step = bottom_optimizer.iterations);
      print('#%d top loss: %f bottom loss: %f' % (top_optimizer.iterations, top_avg_loss.result(), bottom_avg_loss.result()));
      top_avg_loss.reset_states();
      bottom_avg_loss.reset_states();
  
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
