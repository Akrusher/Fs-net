import tensorflow as tf
class Fs_net():
	def __init__(self):
		self.embedding_dim = 128
		self.hidden_dim = 512
		self.n_neurons = 128
		self.n_layers = 2
		self.n_steps = 28
		self.n_inputs = 28
		self.n_outputs = 10
		self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])		

	def bi_lstm(self):
	    # 顺时间循环层的记忆细胞，堆叠了两层
	    lstm_fw1 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
	    lstm_fw2 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
	    lstm_forward = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_fw1,lstm_fw2])
	    # 拟时间循环层的记忆细胞，堆叠了两层
	    lstm_bc1 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
	    lstm_bc2 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
	    lstm_backward = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_bc1,lstm_bc2])
	    # 计算输出和隐状态
	    outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_forward, cell_bw=lstm_backward,inputs=self.X,dtype=tf.float32)
	    # 取到顺时间循环层和拟时间循环层的最后一个隐状态
	    state_forward = states[0][-1][-1]
	    state_backward = states[1][-1][-1]
	    # 把两个隐状态拼接起来。
	    return state_forward+state_backward

	def bi_gru(self,x):
	    gru_forward = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
	    gru_backward = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
	    outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_forward, cell_bw=gru_backward,inputs=x,dtype=tf.float32)
	    return states

	def bi_gru_2(self,x):
	    # 顺时间循环层的记忆细胞，堆叠了两层
	    gru_fw1 = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
	    gru_fw2 = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
	    gru_forward = tf.nn.rnn_cell.MultiRNNCell(cells=[gru_fw1,gru_fw2])
	    # 拟时间循环层的记忆细胞，堆叠了两层
	    gru_bc1 = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
	    gru_bc2 = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
	    gru_backward = tf.nn.rnn_cell.MultiRNNCell(cells=[gru_bc1,gru_bc2])
	    # 计算输出和隐状态
	    outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_forward, cell_bw=gru_backward,inputs=x,dtype=tf.float32)
	    # 取到顺时间循环层和拟时间循环层的最后一个隐状态
	    # state_forward = states[0][-1]
	    # state_backward = states[1][-1]
	    # 把两个隐状态拼接起来。
	    # return state_forward+state_backward
	    return states

	def fs_net(self):
		encoder_output = self.bi_gru(self.X)
		encoder_feats = tf.concate([encoder_output[0][-1], encoder_output[1][-1]],axis=-1)
		decoder_output = self.bi_gru(encoder_feats)
		decoder_feats = tf.concate([decoder_output[0][-1], decoder_output[1][-1]],axis=-1)
		feats = tf.concat([encoder_feats, decoder_feats],axis = -1)
		cls_dense1 = tf.layers.dense(inputs=feats,units= self.n_neurons,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
		cls_dense2 = tf.layers.dense(inputs=cls_dense1,units=self.n_outputs,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
		return cls_dense2
