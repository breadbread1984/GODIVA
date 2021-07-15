#!/usr/bin/python3

import h5py;
import numpy as np;
import tensorflow as tf;

class SampleGenerator(object):  
  def __init__(self, filename):
    f = h5py.File(filename, 'r');
    self.data_train = np.array(f['mnist_gif_train']);
    self.captions_train = np.array(f['mnist_captions_train']);
    self.data_val = np.array(f['mnist_gif_val']);
    self.captions_val = np.array(f['mnist_captions_val']);
    f.close();
  def sample_generator(self, is_trainset = True):
    data = self.data_train if is_trainset else self.data_val;
    caption = self.captions_train if is_trainset else self.captions_val;
    def gen():
      for i in range(data.shape[0]):
        sample = np.transpose(data[i], (0,2,3,1));
        caption = caption[i];
        yield sample, caption;
    return gen;
  def get_trainset(self,):
    return tf.data.Dataset.from_generator(self.sample_generator(True), (tf.float32, tf.int64), (tf.TensorShape([16,64,64,1]), tf.TensorShape([9,]), tf.TensorShape([]))).repeat(-1);
  def get_testset(self):
    return tf.data.Dataset.from_generator(self.sample_generator(False), (tf.float32, tf.int64), (tf.TensorShape([16,64,64,1]), tf.TensorShape([9,]), tf.TensorShape([]))).repeat(-1);

if __name__ == "__main__":

  import cv2;
  generator = SampleGenerator('mnist_single_gif.h5');
  trainset = generator.get_trainset();
  testset = generator.get_testset();
  cv2.namedWindow('sample');
  for sample, caption in trainset:
    print(caption);
    for image in sample:
      cv2.imshow('sample',image.numpy().astype(np.uint8));
      cv2.waitKey(50);
  generator = SampleGenerator('mnist_two_gif.h5');
  trainset = generator.get_trainset();
  testset = generator.get_testset();
