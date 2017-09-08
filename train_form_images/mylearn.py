############################################################################################
# !/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Author  : zhaoqinghui
# Date    : 2016.5.10
# Function: image convert to tfrecords
#############################################################################################

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from PIL import Image
import tf_image

###############################################################################################
image_list = 'debug.txt'
out_tfrecords = './train.tfrecords'
resize_height = 28  # height
resize_width = 28  # width
images = {'0'}
###############################################################################################

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_file(examples_list_file):
    lines = np.genfromtxt(examples_list_file, delimiter=" ", dtype=[('col1', 'S120'), ('col2', 'i8')])
    examples = []
    labels = []
    for example, label in lines:
        examples.append(example)
        labels.append(label)
    return np.asarray(examples), np.asarray(labels), len(lines)


def read_image_info_from_file(filename, resize_height, resize_width):
    print "fileName>>> " + filename
    image = cv2.imread(filename)
    image = cv2.resize(image, (resize_height, resize_width))
    b, g, r = cv2.split(image)
    rgb_image = cv2.merge([r, g, b])
    return rgb_image


def write_tfrecords_file(image_list, out_tfrecords, resize_height, resize_width):
    _examples, _labels, examples_num = load_file(image_list)

    writer = tf.python_io.TFRecordWriter(out_tfrecords)
    for i, [example, label] in enumerate(zip(_examples, _labels)):
        image = read_image_info_from_file(example, resize_height, resize_width)

        print('image: %d, height:%d, width:%d, depth:%d, label: %d' % (
            i, image.shape[0], image.shape[1], image.shape[2], label))

        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()



def create_record(out_tfrecords):
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(out_tfrecords)
    for index, name in enumerate(images):
        class_path = cwd +"/"+ name+"/"
        print name
        i = 0
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name

            image = read_image_info_from_file(img_path, resize_height, resize_width)

            label = name
            print('image: %d, height:%d, width:%d, depth:%d, label: %s' % (
                i, image.shape[0], image.shape[1], image.shape[2], label))

            i += 1
            example = tf.train.Example(features=tf.train.Features(feature=tf_image.read_feature(img_path, label)))

            writer.write(example.SerializeToString())
    writer.close()


def read_record(tfrecord_list_file):

    features = tf_image.read_features(tfrecord_list_file)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)


    init_op = tf.global_variables_initializer()
    resultImg = []
    resultLabel = []
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(21):
            image_eval = image.eval()
            print('step', i, 'training accuracy', image_eval)
            resultLabel.append(label.eval())
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            resultImg.append(image_eval_reshape)
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
        coord.request_stop()
        coord.join(threads)
        sess.close()
    return resultImg, resultLabel

def disp_tfrecords(tfrecord_list_file):
    filename_queue = tf.train.string_input_producer([tfrecord_list_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # print(repr(image))
    height = features['height']
    width = features['width']
    depth = features['depth']
    label = tf.cast(features['label'], tf.int32)
    init_op = tf.global_variables_initializer()
    resultImg = []
    resultLabel = []
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(21):
            image_eval = image.eval()
            print('step', i, 'training accuracy', image_eval)
            resultLabel.append(label.eval())
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            resultImg.append(image_eval_reshape)
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
        coord.request_stop()
        coord.join(threads)
        sess.close()
    return resultImg, resultLabel


def read_tfrecord(filename_queuetemp):
    filename_queue = tf.train.string_input_producer([filename_queuetemp])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # image
    tf.reshape(image, [256, 256, 3])
    # normalize
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # label
    label = tf.cast(features['label'], tf.int32)
    print label
    return image, label


def test():
    write_tfrecords_file(image_list, out_tfrecords, resize_height, resize_width)
    # img, label = disp_tfrecords(out_tfrecords)
    img, label = read_tfrecord(out_tfrecords)
    print label


if __name__ == '__main__':
    # test()
    # disp_tfrecords(out_tfrecords)
    create_record(out_tfrecords)
    # read_tfrecord(out_tfrecords)
