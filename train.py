import tensorflow as tf
from model import Fs_net
from traffic_dataset import dataset

n_steps = 22
n_inputs = 1
batch_size = 16 
X = tf.placeholder(tf.float32, [batch_size, n_steps, n_inputs])
Y = tf.placeholder(tf.int32, [batch_size])	

train_data = dataset.train()
#test_data = dataset.test()

fs_net = Fs_net(X, Y)
loss, logits = fs_net.build_loss()


lr = 1e-3
optimizer  = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)


correct =  tf.nn.in_top_k(logits,Y,1)                        
accuracy = tf.reduce_mean(tf.cast(corret, tf.float32))
tf.summary.scalar("loss",loss)                                                        
tf.summary.scalar("accuracy",accuracy)
summary = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)

n_epoch = 50
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
batch_num = train_data.num_examples // batch_size
model_dir = "./summary"
cnt = 0

with tf.Session() as sess:
	sess.run(init_op)
	train_writer = tf.summary.FileWriter(model_dir,sess.graph)

	for epoch in range(n_epoch):
		for batch in range(batch_num):
			X_batch, y_batch = train_data.next_batch(batch_size)
			_,_,summary_,accuracy_ = sess.run([train_op,loss,summary,accuracy],feed_dict={X:X_batch, Y:y_batch})
			train_writer.add_summary(summary_,cnt)
			print("step: %s accuracy: %s".format(cnt,accuracy_))
			cnt += 1
		if epoch % 5 == 0:
			saver.save(sess,model_dir+"model_{}.ckpt".format(epoch))