#!/usr/bin/python3

import tensorflow as tf;
from models import VQVAE_Trainer;

def main(quantize_type = 'normal'):

  trainer = VQVAE_Trainer(quantize_type = quantize_type);
  trainer.load_weights('./checkpoints/ckpt');
  if quantize_type == 'ema_update':
    trainer.encoder.layers[4].set_trainable(False);
    trainer.encoder.layers[8].set_trainable(False);
  trainer.encoder.save('encoder.h5');
  trainer.decoder.save('decoder.h5');
  
if __name__ == "__main__":

  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s <quantize_type>' % argv[0]);
  assert argv[1] in ['normal', 'ema_update'];
  main(argv[1]);
