#!/usr/bin/python3

from os.path import exists;
import tensorflow as tf;
from models import VQVAE_Trainer;

batch_size = 128;

def main():

  trainer = VQVAE_Trainer();
  if exists('./checkpoints/chkpt'): trainer.load_weights('./checkpoints/ckpt/variables/variables');
  optimizer = tf.keras.optimizers.Adam(1e-5);
  trainer.compile(optimizer = optimizer, loss = {'decoder':, 'diff': });
