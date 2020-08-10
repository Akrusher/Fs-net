import random
import numpy as np
class dataset:
	class train:
		def __init__(self):
			self.batch_ptr = 0
			self.label_path = "train_pcap_length.txt" 
			self.label_file = open(self.label_path,"r").readlines()
			self.num_examples = len(self.label_file)
			self.n_steps = 256
			self.n_inputs = 1
			self.label2idx = {"iqiyi":0,"taobao":1,"weibo":2,"weixin":3}
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
				content = line.strip().split()
				y_batch[idx] = self.label2idx[content[0].split("_")[0]]
				for i in range(1, 257):

					x_batch[idx][i-1][0] = int(content[i])*1.0
				idx += 1
			self.batch_ptr += 1
			return x_batch, y_batch

		def set_batch_ptr(self,batch_ptr):
			self.batch_ptr = batch_ptr


	class test:
		def __init__(self):
			self.batch_ptr = 0
			self.label_path = "test_pcap_length.txt" 
			self.label_file = open(self.label_path,"r").readlines()
			self.num_examples = len(self.label_file)
			self.n_steps = 256
			self.n_inputs = 1
			self.label2idx = {"iqiyi":0,"taobao":1,"weibo":2,"weixin":3}
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
				content = line.strip().split()
				y_batch[idx] = self.label2idx[content[0].split("_")[0]]
				for i in range(1, 257):
					x_batch[idx][i-1][0] = int(content[i])*1.0
				idx += 1
			self.batch_ptr += 1
			return x_batch, y_batch

		def set_batch_ptr(self,batch_ptr):
			self.batch_ptr = batch_ptr


