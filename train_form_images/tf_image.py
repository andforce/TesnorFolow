import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from PIL import Image


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_image_info_from_file(filename):
    print "fileName>>> " + filename
    image = cv2.imread(filename)
    b, g, r = cv2.split(image)
    rgb_image = cv2.merge([r, g, b])
    return rgb_image

def read_feature(img_path, img_label):
    image = read_image_info_from_file(img_path)
    height, width, depth = image.shape
    image_raw = image.tostring()
    feature = {
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(depth),
        'label': _bytes_feature(img_label),
        'image_raw': _bytes_feature(image_raw)
    }
    return feature

def read_features(tfrecords):
    filename_queue = tf.train.string_input_producer([tfrecords])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })
    return features