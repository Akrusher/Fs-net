import tensorflow as tf
class Fs_net():
	def __init__(self):
		self.embedding_dim = 128
		self.hidden_dim = 512
		self.n_neurons = 512
		self.n_layers = 2
		self.n_steps = 28
		self.n_inputs = 28
		self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])		
		self.cell = "GRU"
		
	def cell_selected(self):
		if cell == "RNN":
			rnn_cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu) for layer in range(n_layers)]
        	multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)
        	outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, self.X, dtype=tf.float32)
        	return tf.concat(axis=1, values=states)
		
		elif cell == "LSTM":
			lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons) for layer in range(n_layers)]
        	multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        	outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
			return states[-1][1]

		elif cell == "GRU":
			gru_cells = [tf.nn.rnn_cell.GRUCell(num_units = self.n_neurons) for layer in range(self.n_layers)]
			multi_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
			outputs, states = tf.nn.dynamic_rnn(multi_cell, self.X, dtype.float32)
			return states[-1]
