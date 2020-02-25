import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.contrib.slim as slim

from model import mobilenet_v2
from model import mobilenet

def conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', activation_fn=nn.relu, bn=True):
    net = slim.conv2d(inputs, num_outputs, kernel_size, stride, padding, activation_fn=None)
    if bn:
        net = slim.batch_norm(net)
    net = activation_fn(net)
    return net
    
def conv2d_transpose(inputs, num_outputs, kernel_size, stride=1, padding='SAME', activation_fn=nn.relu, bn=True):
    net = slim.conv2d_transpose(inputs, num_outputs, kernel_size, stride, padding, activation_fn=None)
    if bn:
        net = slim.batch_norm(net)
    net = activation_fn(net)
    return net

def upsample(inputs, size):
    shape = inputs.get_shape().as_list()
    return tf.image.resize_images(inputs, [shape[1] * size, shape[2] * size])

def build_mobilenet_v2_unet(inputs, num_classes):
    logits, end_points = mobilenet_v2.mobilenet_base(inputs)
    with tf.variable_scope('UNet') as scope:

        net = conv2d(logits, 320, 3)
        
        net = tf.concat([
            tf.get_default_graph().get_tensor_by_name('MobilenetV2/expanded_conv_16/output:0'),
            net
        ], -1)
        net = conv2d(net, 320, 3)
        net = conv2d(net, 192, 3)

        net = conv2d_transpose(net, 192, 3, stride=2)
        net = tf.concat([
            tf.get_default_graph().get_tensor_by_name('MobilenetV2/expanded_conv_12/output:0'),
            tf.get_default_graph().get_tensor_by_name('MobilenetV2/expanded_conv_11/output:0'),
            net
        ], -1)
        net = conv2d(net, 192, 3)
        net = conv2d(net, 64, 3)

        net = conv2d_transpose(net, 64, 3, stride=2)
        net = tf.concat([
            tf.get_default_graph().get_tensor_by_name('MobilenetV2/expanded_conv_5/output:0'),
            tf.get_default_graph().get_tensor_by_name('MobilenetV2/expanded_conv_4/output:0'),
            net
        ], -1)
        net = conv2d(net, 64, 3)
        net = conv2d(net, 32, 3)

        net = conv2d_transpose(net, 24, 3, stride=2)
        net = tf.concat([
            tf.get_default_graph().get_tensor_by_name('MobilenetV2/expanded_conv_2/output:0'),
            net
        ], -1)
        net = conv2d(net, 24, 3)
        net = conv2d(net, 16, 3)

        net = conv2d_transpose(net, 16, 3, stride=2)
        net = tf.concat([
            tf.get_default_graph().get_tensor_by_name('MobilenetV2/expanded_conv/output:0'),
            net
        ], -1)
        net = conv2d(net, 16, 3)
        net = conv2d(net, 8, 3)

        net = conv2d_transpose(net, 8, 3, stride=2)
        net = conv2d(net, 8, 3)
        # net = slim.conv2d(net, num_classes, 1, activation_fn=nn.softmax)
        net = slim.conv2d(net, num_classes, 1, activation_fn=nn.sigmoid)

    return net
    
def load_pretrained(sess):
    variables_to_restore = {v.name.split(":")[0]: v
                        for v in tf.get_collection(
                            tf.GraphKeys.GLOBAL_VARIABLES)}

    variables_to_restore = {
        v: variables_to_restore[v] for
        v in variables_to_restore if v.find('MobilenetV2') != -1}

    saver = tf.train.Saver(var_list=variables_to_restore)
    saver.restore(sess, 'mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt')