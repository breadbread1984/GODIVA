#!/usr/bin/python3

import numpy as np;
import cv2;
import tensorflow as tf;
from models import Quantize;

def main(img_path):
  img = cv2.imread(img_path);
  if img is None:
    raise Exception('invalid image path');
  encoder = tf.keras.models.load_model('encoder.h5', custom_objects = {'Quantize': Quantize});
  decoder = tf.keras.models.load_model('decoder.h5');
  resized = cv2.resize(img, (256, 256)) / 255.;
  normalized = (resized - tf.reshape([0.5,0.5,0.5], (1,1,-1))) / tf.reshape([0.5,0.5,0.5], (1,1,-1));
  inputs = np.expand_dims(normalized, axis = 0);
  quantized_t, cluster_index_t, diff_t, quantized_b, cluster_index_b, diff_b = encoder(inputs);
  recon = decoder([quantized_t, quantized_b]);
  recon_img = tf.squeeze(recon * 128., axis = 0).numpy() * np.reshape([0.5,0.5,0.5], (1, 1, -1)) + np.reshape([0.5,0.5,0.5], (1, 1, -1));
  recon_img = 255. * recon_img;
  cv2.imshow('reconstructed', recon_img.astype(np.uint8));
  cv2.waitKey();

if __name__ == "__main__":
  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s <image>' % argv[0]);
    exit(1);
  main(argv[1]);
