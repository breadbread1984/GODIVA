#!/usr/bin/python3

from re import search;
from os import listdir;
from os.path import join;
import tensorflow as tf;

def parse_function_generator(output_size=(256, 256)):
  def parse_function(serialized_example):
    feature = tf.io.parse_single_example(
      serialized_example,
      features={
        'image/encoded': tf.FixedLenFeature((), dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature((), dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature((), dtype=tf.string, default_value='')
      }
    );
    image = tf.io.decode_jpeg(feature['image/encoded']);
    batched_image = tf.expand_dims(image, axis=0);
    batched_resized = tf.image.resize(batched_image, (output_size[0], output_size[1]), method=tf.image.ResizeMethod.BILINEAR);
    image = tf.squeeze(batched_resized, axis = 0);
    image = tf.cast(image, dtype=tf.float32) - tf.reshape([123.68, 116.78, 103.94], (1, 1, -1));
    image = image / 128.;
    label = tf.cast(feature['image/class/label'], dtype=tf.int32);
    return image, (image, label);
  return parse_function;

def load_dataset(directory):
  filenames = [join(directory, filename) for filename in listdir(directory) if search('-of-', filename)];
  filenames = sorted(filenames);
  dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(buffer_size=len(filenames)).flat_map(tf.data.TFRecordDataset);
  return dataset;

if __name__ == "__main__":
  
