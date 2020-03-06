import os, sys, time

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2

from model import unet

net_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

network = unet.build_mobilenet_v2_unet(net_input, 2)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

saver = tf.train.Saver()
model_checkpoint_name = 'checkpoints/model.ckpt'

saver.restore(sess, model_checkpoint_name)

for i in range(1000,3889):
    img = cv2.imread('/home/wan/data/frame_4/{}.jpg'.format(i))
    # cv2.imshow("Camera", img)
    input_image = np.expand_dims(cv2.resize(img, (224,224)), 0) / 255.
    output_image = sess.run(network, feed_dict={net_input: input_image})
    seg = np.concatenate((output_image[0],np.zeros(shape=(224,224,1))), axis=-1) * 255.
    seg = cv2.resize(seg, (640, 480))

    # cv2.imshow("Seg", seg)

    dst = (0.5 * img + 0.5 * seg) / 255.
    cv2.imshow('', dst)
    if cv2.waitKey(1) == 27:
        break

sess.close()