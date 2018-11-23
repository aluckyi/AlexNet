import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from Net import AlexNet
import config as cfg
from scipy.misc import imread, imresize


if __name__ == '__main__':
    net = AlexNet()
    src = imread('test2.jpg',  mode='RGB')
    image = imresize(src, (cfg.image_H, cfg.image_W))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.load_weights(sess)

        probs, label_preds = sess.run((net.probs, net.label_preds),
                                      feed_dict={net.images: [image], net.is_training: False})
        label = cfg.classes[label_preds[0]]
        prob = probs[0][label_preds[0]]
        message = label + ':' + str(prob)

    plt.imshow(src)
    plt.text(5, 30, message, fontsize=18, color='y')
    plt.show()
