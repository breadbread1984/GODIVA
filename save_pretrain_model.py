#!/usr/bin/python3

import tensorflow as tf;
from models import VQVAE_Trainer;

def main():

  trainer = VQVAE_Trainer();
  trainer.load_weights('./checkpoints/ckpt');
  trainer.encoder.layers[4].set_trainable(False);
  trainer.encoder.layers[8].set_trainable(False);
  trainer.encoder.save('encoder.h5');
  trainer.decoder.save('decoder.h5');
  
if __name__ == "__main__":
  main();
