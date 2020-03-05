import os,time,cv2, sys, math

import tensorflow as tf
import cv2
import numpy as np
import random

from model import unet

num_classes = 1
num_epochs = 20
batch_size = 1
is_continue_training = False


sess = tf.Session()

net_input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
net_output = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, num_classes])
network = unet.build_mobilenet_v2_unet(net_input, num_classes)

variables_to_train = {
    v.name: v for v in tf.trainable_variables()
}
variables_to_train = {
    v: variables_to_train[v]
    for v in variables_to_train if v.find('UNet') != -1
}

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=network, labels=net_output, pos_weight=5))
# opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=variables_to_train)
opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, var_list=variables_to_train)

sess.run(tf.global_variables_initializer())
unet.load_pretrained(sess)

saver = tf.train.Saver()

model_checkpoint_name = 'checkpoints/model.ckpt'

if is_continue_training:
    print('Restore latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

print('Loading the data ...')

# train_input_names = []
# train_output_names = []
train_inputs = []
train_outputs = []
with open('/content/data/train.txt') as f:
    cnt = 0
    for line in f:
        # train_input_names.append(line.split(' ')[0])
        # train_output_names.append(line.split(' ')[1])
        train_inputs.append(cv2.imread(line.split(' ')[0]))
        gt = np.expand_dims(cv2.imread(line.split(' ')[1], cv2.IMREAD_GRAYSCALE), -1)
        gt = np.concatenate((gt,np.zeros(shape=(224,224,1))), axis=-1)
        try:
            json_file = line.split(' ')[0].split('.')[0] + '.json'
            with open(json_file) as f:
                meta = json.load(f)
                for shape in meta['shapes']:
                    p1 = shape['points'][0]
                    for i in range(1,len(shape['points'])):
                        p2 = shape['points'][i]
                        gt = cv2.line(gt, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,255), 2)
                        p1 = p2
            cnt += 1
        except:
            pass
        train_outputs.append(gt)
    print('Number of images with dash lines:', cnt)

# Start the training here
for epoch in range(num_epochs):

    current_losses = []

    cnt=0

    # id_list = np.random.permutation(len(train_input_names))
    id_list = np.random.permutation(len(train_inputs))
    

    num_iters = int(np.floor(len(id_list) / batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):

        input_image_batch = []
        output_image_batch = []

        for j in range(batch_size):
            index = i*batch_size + j
            id = id_list[index]
            input_image = train_inputs[id] / 255.
            output_image = train_outputs[id] / 255.
            # input_image = cv2.imread(train_input_names[id]) / 255.
            # output_image = cv2.imread(train_output_names[id], cv2.IMREAD_GRAYSCALE) / 255.
            # output_image = np.expand_dims(output_image, -1)

            input_image_batch.append(input_image)
            output_image_batch.append(output_image)

        input_image_batch = np.array(input_image_batch)
        output_image_batch = np.array(output_image_batch)

        # Do the training
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)
        cnt = cnt + batch_size
        if cnt % 128 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            print(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)

    # Create directories if needed
    if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
        os.makedirs("%s/%04d"%("checkpoints",epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)
