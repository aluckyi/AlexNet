import os
import numpy as np
import tensorflow as tf

import config as cfg
import matplotlib.pyplot as plt


class Data(object):
    def __init__(self, session, is_training=False):
        self.batch_size = cfg.batch_size
        self.image_size = (cfg.image_H, cfg.image_W, cfg.image_C)
        self.data_set_path = cfg.data_set_path
        self.train_list_name = cfg.train_list_name
        self.val_list_name = cfg.val_list_name
        self.test_list_name = cfg.test_list_name

        self.session = session
        self.is_training = is_training

        if self.is_training:
            train_names, train_labels = self.parse_list_file(
                os.path.join(self.data_set_path, self.train_list_name)
            )
            self.train_dataset = tf.data.Dataset.from_tensor_slices((train_names, train_labels))
            self.train_dataset = self.train_dataset.shuffle(100).map(self.map_function).batch(self.batch_size,
                                                                                              drop_remainder=True)
            self.train_iterator = self.train_dataset.make_initializable_iterator()
            self.train_next = self.train_iterator.get_next()
            self.session.run(self.train_iterator.initializer)

            val_names, val_labels = self.parse_list_file(
                os.path.join(self.data_set_path, self.val_list_name)
            )
            self.val_dataset = tf.data.Dataset.from_tensor_slices((val_names, val_labels))
            self.val_dataset = self.val_dataset.map(self.map_function).batch(self.batch_size,
                                                                             drop_remainder=True)
            self.val_iterator = self.val_dataset.make_initializable_iterator()
            self.val_next = self.val_iterator.get_next()
            self.session.run(self.val_iterator.initializer)

        else:
            test_names, test_labels = self.parse_list_file(
                os.path.join(self.data_set_path, self.test_list_name)
            )
            self.test_dataset = tf.data.Dataset.from_tensor_slices((test_names, test_labels))
            self.test_dataset = self.test_dataset.map(self.map_function).batch(self.batch_size,
                                                                               drop_remainder=True)
            self.test_iterator = self.test_dataset.make_initializable_iterator()
            self.test_next = self.test_iterator.get_next()
            self.session.run(self.test_iterator.initializer)

    def reinitialize(self):
        if self.is_training:
            self.session.run(self.train_iterator.initializer)
            self.session.run(self.val_iterator.initializer)
        else:
            self.session.run(self.test_iterator.initializer)

    def get_train_batch(self):
        return self.session.run(self.train_next)

    def get_val_batch(self):
        return self.session.run(self.val_next)

    def get_test_batch(self):
        return self.session.run(self.test_next)

    def map_function(self, name, label):
        file_path = tf.strings.join([self.data_set_path, name], separator='/')
        image_string = tf.read_file(file_path)
        # image_decoded = tf.image.decode_image(image_string)
        image = tf.image.decode_jpeg(image_string, channels=3)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [self.image_size[0], self.image_size[1]])
        return image, label

    def parse_list_file(self, file_path):
        names, labels = [], []
        with open(file_path, mode='r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            names.append(line[0])
            labels.append(int(line[1]))
        return names, labels


if __name__ == '__main__':
    with tf.Session() as sess:
        # data = Data(sess, is_training=True)
        #
        # image, label = data.get_train_batch()
        # print(image[0])
        # # print(label)
        # print(image.shape)
        # print(label.shape)
        #
        # data.reinitialize()
        # image, label = data.get_train_batch()
        # # print(image)
        # # print(label)
        # print(image.shape)
        # print(label.shape)

        data = Data(sess, is_training=False)

        image, label = data.get_test_batch()
        print(image[0])
        print(image.shape)
        print(label.shape)
