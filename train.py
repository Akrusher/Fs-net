import tensorflow as tf
from model import Fs_net
#from traffic_dataset import dataset
from dataset_pcap_length import dataset
import numpy as np

n_steps = 256
n_inputs = 1
batch_size = 8 
X = tf.placeholder(tf.float32, [batch_size, n_steps, n_inputs])
Y = tf.placeholder(tf.int32, [batch_size])    

train_data = dataset.train()
test_data = dataset.test()

fs_net = Fs_net(X, Y)
loss, logits = fs_net.build_loss()


lr = 1e-3
optimizer  = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)


correct =  tf.nn.in_top_k(logits,Y,1)                        
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
tf.summary.scalar("loss",loss)                                                        
#tf.summary.scalar("accuracy",accuracy)
summary = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)

n_epoch = 1500
batch_num = train_data.num_examples // batch_size
training_steps_per_epoch = 3
model_dir = "./summary/"
cnt = 0

with tf.Session() as sess:
    sess.run(init_op)
    train_writer = tf.summary.FileWriter(model_dir,sess.graph)

    for epoch in range(n_epoch):
        for batch in range(batch_num):
            X_batch, y_batch = train_data.next_batch(batch_size)
            _,_,summary_,train_acc = sess.run([train_op,loss,summary,accuracy],feed_dict={X:X_batch, Y:y_batch})
            train_writer.add_summary(summary_,cnt)
            cnt += 1
            if cnt % training_steps_per_epoch == 0:
                print("step: {} train_accuracy: {}".format(cnt, train_acc))
        steps_per_epoch = test_data.num_examples // batch_size
        correct_cnt = 0
        num_test_examples = steps_per_epoch * batch_size
        for test_steps in range(steps_per_epoch):
            X_test, y_test = test_data.next_batch(batch_size)
            logits_ = sess.run(logits,feed_dict={X:X_test, Y:y_test})
            prediction = np.argmax(logits_,1)
            correct_cnt += np.sum(prediction == y_test)
        acc = correct_cnt * 1.0 / num_test_examples
        print("step: {} test_accuracy: {}".format(cnt, acc))
        if epoch % 2 == 0:
            saver.save(sess,model_dir+"model_{}.ckpt".format(epoch))
