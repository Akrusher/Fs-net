import tensorflow as tf 
import numpy as np
import cv2
import math
from dataset_pcap_length import dataset
from model import Fs_net
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

n_steps = 256
n_inputs = 1
batch_size = 8 
X = tf.placeholder(tf.float32, [batch_size, n_steps, n_inputs])
Y = tf.placeholder(tf.int32, [batch_size])  

test_data = dataset.test()

fs_net = Fs_net(X, Y)
loss, logits = fs_net.build_loss()

sess = tf.Session()
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint('summary/')
sess.run(tf.global_variables_initializer())
saver.restore(sess, checkpoint)

steps_per_epoch = test_data.num_examples // batch_size
num_test_examples = steps_per_epoch * batch_size
correct_cnt = 0
for test_steps in range(steps_per_epoch):
    X_test, y_test = test_data.next_batch(batch_size)
    tic = time.time()
    logits_ = sess.run(logits,feed_dict={X:X_test, Y:y_test})
    prediction = np.argmax(logits_,1)
    correct_cnt += np.sum(prediction == y_test)
    toc = time.time() 
    print("model take {}".format(toc-tic))
acc = correct_cnt * 1.0 / num_test_examples
print("test_accuracy: {}".format(acc))


