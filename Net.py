import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.slim as slim

import config as cfg


class AlexNet(object):
    def __init__(self):
        self.class_num = cfg.class_num
        self.train_set_mean = cfg.train_set_mean
        self.keep_prob = cfg.keep_prob
        self.initial_weights_path = cfg.initial_weights_path
        self.weights_path = cfg.weights_path

        self.regularizer = tfc.layers.l2_regularizer(cfg.reg_scale)

        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.images = tf.placeholder(tf.float32, [None, cfg.image_H, cfg.image_W, cfg.image_C], name="images")
        self.labels = tf.placeholder(tf.int32, [None], name="labels")

        mean = tf.constant(self.train_set_mean, dtype=tf.float32, shape=[1, 1, 1, 3])
        self.images = self.images - mean

        # build network
        self.conv_layers()
        self.fc_layers()

        # loss and probs
        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.fc8)
        tf.losses.add_loss(loss)
        self.loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', self.loss)

        self.probs = tf.nn.softmax(self.fc8, name="probs")
        self.label_preds = tf.argmax(self.probs, axis=1, name="label_preds", output_type=tf.int32)
        self.correct_labels = tf.equal(self.label_preds, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_labels, tf.float32))
        self.correct_times_in_batch = tf.reduce_sum(tf.cast(self.correct_labels, tf.int32))

        # restorer and saver
        exclude = ['fc8/w', 'fc8/b']
        self.variable_to_restore = slim.get_variables_to_restore(exclude=exclude)
        self.initial_restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=1)

        self.restorer = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)

    def conv_layers(self):
        # the 1st convolution layer
        self.conv1 = self.convLayer(self.images, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = self.LRN(self.conv1, 5, 1e-04, 0.75, "norm1")
        self.pool1 = self.maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")
        # the 2nd convolution layer
        self.conv2 = self.convLayer(self.pool1, 5, 5, 1, 1, 256, "conv2", groups=2)
        lrn2 = self.LRN(self.conv2, 5, 1e-04, 0.75, "lrn2")
        self.pool2 = self.maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")
        # the 3rd convolution layer
        self.conv3 = self.convLayer(self.pool2, 3, 3, 1, 1, 384, "conv3")
        # the 4th convolution layer
        self.conv4 = self.convLayer(self.conv3, 3, 3, 1, 1, 384, "conv4", groups=2)
        # the 5th convolution layer
        self.conv5 = self.convLayer(self.conv4, 3, 3, 1, 1, 256, "conv5", groups=2)
        self.pool5 = self.maxPoolLayer(self.conv5, 3, 3, 2, 2, "pool5", "VALID")

    def fc_layers(self):
        self.pool5_flat = tfc.layers.flatten(self.pool5)
        # the 6th fully connected layer
        self.fc6 = self.fcLayer(self.pool5_flat, 4096, "fc6")
        self.drop6 = tfc.layers.dropout(self.fc6, keep_prob=self.keep_prob,
                                        is_training=self.is_training, scope="dropout6")
        # the 7th fully connected layer
        self.fc7 = self.fcLayer(self.drop6, 4096, "fc7")
        self.drop7 = tfc.layers.dropout(self.fc7, keep_prob=self.keep_prob,
                                        is_training=self.is_training, scope="dropout7")

        # the 8th fully connected layer
        self.fc8 = self.fcLayer_(self.drop7, self.class_num, "fc8")

    def convLayer(self, x, kHeight, kWidth, strideX, strideY, featureNum, name, padding="SAME", groups=1):
        channel = int(x.get_shape()[-1])

        conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, strideY, strideX, 1], padding=padding)

        with tf.variable_scope(name):
            w = tf.get_variable("w", dtype=tf.float32, shape=[kHeight, kWidth, channel / groups, featureNum])
            b = tf.get_variable("b", dtype=tf.float32, shape=[featureNum])
            tf.losses.add_loss(self.regularizer(w))

            xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)
            wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)

            featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
            mergeFeatureMap = tf.concat(values=featureMap, axis=3)
            out = tf.nn.bias_add(mergeFeatureMap, b)
            # return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()))
        return tf.nn.relu(out)

    def fcLayer_(self, x, num_outputs, name):
        channel = int(x.get_shape()[-1])

        with tf.variable_scope(name):
            w = tf.get_variable("w", dtype=tf.float32, shape=[channel, num_outputs],
                                initializer=tf.truncated_normal_initializer(0.0, 0.01))
            b = tf.get_variable("b", dtype=tf.float32, shape=[num_outputs],
                                initializer=tf.truncated_normal_initializer(0.0, 0.01))
            tf.losses.add_loss(self.regularizer(w))
        return tf.matmul(x, w) + b

    def fcLayer(self, x, num_outputs, name):
        out = self.fcLayer_(x, num_outputs, name)
        return tf.nn.relu(out)

    def LRN(self, x, radius, alpha, beta, name=None, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                                  beta=beta, bias=bias, name=name)

    def maxPoolLayer(self, x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
        return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                              strides=[1, strideX, strideY, 1],
                              padding=padding, name=name)

    def load_initial_weights(self, session):
       self.initial_restorer.restore(session, self.initial_weights_path)

    def load_weights(self, session):
        self.restorer.restore(session, self.weights_path)

    def save_weights(self, session):
        self.saver.save(session, self.weights_path)


class VGG16(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    net = AlexNet()
    for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(x)
    print('*' * 88)
    for x in tf.trainable_variables():
        print(x)

    with tf.Session() as sess:
        flag = sess.run(net.is_training, feed_dict={net.is_training: True})
        print(flag)

