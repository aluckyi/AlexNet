import os
import math
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

from Net import AlexNet
from Data import Data
import config as cfg


class Solver(object):
    def __init__(self, net):
        self.net = net

        self.classes = cfg.classes
        self.class_num = cfg.class_num
        self.batch_size = cfg.batch_size

        self.config_dir = cfg.config_dir

        self.initial_learning_rate = cfg.initial_learning_rate
        self.decay_steps = cfg.decay_steps
        self.decay_rate = cfg.decay_rate
        self.staircase = cfg.staircase
        self.num_epoch = cfg.num_epoch
        self.summary_path = cfg.summary_path
        self.summary_steps = cfg.summary_steps
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(logdir=self.summary_path)

    def train(self, is_validating=False):
        self.global_step = tf.get_variable(
            name='global_step', shape=[],
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step,
            self.decay_steps, self.decay_rate,
            self.staircase, name='learning_rate'
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.net.loss, global_step=self.global_step
        )
        self.ema = tf.train.ExponentialMovingAverage(0.9)
        self.ema_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.ema_op)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # load initial model weights
            self.net.load_initial_weights(sess)
            print('loading the initial model weights...')

            step = 0
            max_accuracy = 0.0
            data = Data(sess, is_training=True)
            if is_validating:
                losses = [[], []]
                accuracies = [[], []]
                self.view_bar('training...', 0, self.num_epoch)
            for epoch_no in range(self.num_epoch):
                # train model
                while True:
                    try:
                        train_images, train_labels = data.get_train_batch()
                        feed_dict = {self.net.images: train_images,
                                     self.net.labels: train_labels,
                                     self.net.is_training: True}
                        summary, _ = sess.run((self.summary_op, self.train_op), feed_dict=feed_dict)
                        if step % self.summary_steps == 0:
                            self.summary_writer.add_summary(summary, step)
                        step += 1
                    except tf.errors.OutOfRangeError:
                        break
                # get losses and accuracies of train and validation
                if is_validating:
                    train_loss, train_accuracy, val_loss, val_accuracy = self.get_train_val_loss_accuracy(sess, data)
                    losses[0].append(train_loss)
                    losses[1].append(val_loss)
                    accuracies[0].append(train_accuracy)
                    accuracies[1].append(val_accuracy)
                    '''
                    template = 'Epoch: {}, train_loss:{:.6f}, train_accuracy:{:6f},' \
                               'val_loss:{:.6f}, val_accuracy:{:.6f}'
                    print(template.format(epoch_no, train_loss, train_accuracy, val_loss, val_accuracy))
                    '''
                    # save trained weights
                    if val_accuracy > max_accuracy:
                        max_accuracy = val_accuracy
                        self.net.save_weights(sess)
                data.reinitialize()

                # display train process
                self.view_bar('training...', epoch_no + 1, self.num_epoch)

            # show visualize train / val loss and train / val accuracy
            if is_validating:
                self.visualize_loss_accuracy_curve(losses, accuracies)

            # save trained weights
            if not is_validating:
                self.net.save_weights(sess)
            # save configurations
            self.save_cfg()

    def test(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # load model weights
            self.net.load_weights(sess)
            print('loading the model weights...')

            data = Data(sess, is_training=False)
            total_batches_in_test_set = 0.0
            total_loss_in_test_set = 0.0
            total_correct_times_in_test_set = 0.0
            while True:
                try:
                    test_images, test_labels = data.get_test_batch()
                    feed_dict = {self.net.images: test_images,
                                 self.net.labels: test_labels,
                                 self.net.is_training: False}
                    loss, correct_times_in_batch = sess.run(
                        (self.net.loss, self.net.correct_times_in_batch),
                        feed_dict=feed_dict
                    )
                    total_batches_in_test_set += 1
                    total_loss_in_test_set += (loss * self.batch_size)
                    total_correct_times_in_test_set += correct_times_in_batch
                except tf.errors.OutOfRangeError:
                    break
            test_loss = total_loss_in_test_set / float(total_batches_in_test_set * self.batch_size)
            test_accuracy = total_correct_times_in_test_set / float(total_batches_in_test_set * self.batch_size)
            template = 'test_loss:{:.6f}, test_accuracy:{:6f},'
            print(template.format(test_loss, test_accuracy))

    def get_train_val_loss_accuracy(self, session, data):
        data.reinitialize()
        # get train loss and accuracy
        total_batches_in_train_set = 0.0
        total_loss_in_train_set = 0.0
        total_correct_times_in_train_set = 0.0
        while True:
            try:
                train_images, train_labels = data.get_train_batch()
                feed_dict = {self.net.images: train_images,
                             self.net.labels: train_labels,
                             self.net.is_training: False}
                loss, correct_times_in_batch = session.run(
                    (self.net.loss, self.net.correct_times_in_batch),
                    feed_dict=feed_dict
                )
                total_batches_in_train_set += 1
                total_loss_in_train_set += (loss * self.batch_size)
                total_correct_times_in_train_set += correct_times_in_batch
            except tf.errors.OutOfRangeError:
                break
        train_loss = total_loss_in_train_set / float(total_batches_in_train_set * self.batch_size)
        train_accuracy = total_correct_times_in_train_set / float(total_batches_in_train_set * self.batch_size)

        # get validation loss and accuracy
        total_batches_in_val_set = 0.0
        total_loss_in_val_set = 0.0
        total_correct_times_in_val_set = 0.0
        while True:
            try:
                val_images, val_labels = data.get_val_batch()
                feed_dict = {self.net.images: val_images,
                             self.net.labels: val_labels,
                             self.net.is_training: False}
                loss, correct_times_in_batch = session.run(
                    (self.net.loss, self.net.correct_times_in_batch),
                    feed_dict=feed_dict
                )
                total_batches_in_val_set += 1
                total_loss_in_val_set += (loss * self.batch_size)
                total_correct_times_in_val_set += correct_times_in_batch
            except tf.errors.OutOfRangeError:
                break
        val_loss = total_loss_in_val_set / float(total_batches_in_val_set * self.batch_size)
        val_accuracy = total_correct_times_in_val_set / float(total_batches_in_val_set * self.batch_size)
        return train_loss, train_accuracy, val_loss, val_accuracy

    def save_cfg(self):
        with open(os.path.join(self.config_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0] != '_' and key != 'os':
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def visualize_loss_accuracy_curve(self, losses, accuracies):
        plt.subplot(2, 1, 1)
        plt.title('loss')
        plt.plot(losses[0], '-o', label='train')
        plt.plot(losses[1], '-o', label='val')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15, 12)

        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(accuracies[0], '-o', label='train')
        plt.plot(accuracies[1], '-o', label='val')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15, 12)
        plt.show()

    def view_bar(self, message, num, total):

        rate = num / total
        rate_num = int(rate * 40)
        rate_nums = math.ceil(rate * 100)
        r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
        sys.stdout.write(r)
        sys.stdout.flush()


if __name__ == '__main__':
    is_training = False

    net = AlexNet()
    solver = Solver(net)
    if is_training:
        solver.train(is_validating=True)
    else:
        solver.test()

