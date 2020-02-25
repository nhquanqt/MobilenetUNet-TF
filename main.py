import os, sys, time

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2

from model import unet

net_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

network = unet.build_mobilenet_v2_unet(net_input, 2)

variables_to_restore = {v.name.split(":")[0]: v
                        for v in tf.get_collection(
                            tf.GraphKeys.GLOBAL_VARIABLES)}

variables_to_restore = {
    v: variables_to_restore[v] for
    v in variables_to_restore if v.find('MobilenetV2') != -1}

# print("*** Restore variables ***")
# for var in variables_to_restore:
#     print(var)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# saver = tf.train.Saver(var_list=variables_to_restore)
# saver.restore(sess, 'mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt')


# print("*** Trainable variables ***")
# for var in tf.trainable_variables():
#     print(var)

# slim.model_analyzer.analyze_vars(model_vars, print_info=True)

for n in tf.get_default_graph().as_graph_def().node:
    print(n.name)

# t_now = time.time()
# for _ in range(100):
#     input_image = np.random.rand(1,224,224,3)
#     output_image = sess.run(network, feed_dict={net_input:input_image})
#     print(output_image.shape ,1./(time.time() - t_now))
#     t_now = time.time()

sess.close()