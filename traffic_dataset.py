import random
import numpy as np
class dataset:
	class train:
		def __init__(self):
			self.batch_ptr = 0
			self.label_path = "NIMS_train.arff" 
			self.label_file = open(self.label_path,"r").readlines()
			self.num_examples = len(self.label_file)
			self.n_steps = 22
			self.n_inputs = 1
			self.label2idx = {"TELNET":0,"FTP":1,"HTTP":2,"DNS":3,"lime":4,"localForwarding":5,"remoteForwarding":6,"scp":7,"sftp":8,"x11":9,"shell":10}
			random.shuffle(self.label_file)

		def next_batch(self, batch_size, ptr = None):
			x_batch = np.zeros([batch_size,self.n_steps,self.n_inputs])
			y_batch = np.zeros([batch_size],np.int32)
			if ptr == None:
				ptr = self.batch_ptr * batch_size
			if ptr + batch_size >= self.num_examples:
				self.set_batch_ptr(0)
				ptr = self.batch_ptr * batch_size
			label_lines = self.label_file[ptr:ptr+batch_size]
			idx = 0
			for line in label_lines:
				content = line.strip().split(',')
				y_batch[idx] = self.label2idx[content[len(content) - 1]]
				for i in range(len(content) - 1):
					x_batch[idx][i][0] = int(content[i])
				idx += 1
			self.batch_ptr += 1
			return x_batch, y_batch

		def set_batch_ptr(self,batch_ptr):
			self.batch_ptr = batch_ptr


	class test:
		def __init__(self):
			self.batch_ptr = 0
			self.label_path = "NIMS_test.arff" 
			self.label_file = open(self.label_path,"r").readlines()
			self.num_examples = len(self.label_file)
			self.n_steps = 22
			self.n_inputs = 1
			self.label2idx = {"TELNET":0,"FTP":1,"HTTP":2,"DNS":3,"lime":4,"localForwarding":5,"remoteForwarding":6,"scp":7,"sftp":8,"x11":9,"shell":10}
			random.shuffle(self.label_file)

		def next_batch(self, batch_size, ptr = None):
			x_batch = np.zeros([batch_size,self.n_steps,self.n_inputs])
			y_batch = np.zeros([batch_size],np.int32)
			if ptr == None:
				ptr = self.batch_ptr * batch_size
			if ptr + batch_size >= self.num_examples:
				self.set_batch_ptr(0)
				ptr = self.batch_ptr * batch_size
			label_lines = self.label_file[ptr:ptr+batch_size]
			idx = 0
			for line in label_lines:
				content = line.strip().split(',')
				y_batch[idx] = label2idx[content[len(content) - 1]]
				for i in range(len(content) - 1):
					x_batch[idx][i][0] = int(content[i])
				idx += 1
			self.batch_ptr += 1
			return x_batch, y_batch

		def set_batch_ptr(self,batch_ptr):
			self.batch_ptr = batch_ptr


