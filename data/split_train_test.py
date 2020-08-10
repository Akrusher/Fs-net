import random
f1 = open("pcap_length.txt","r")
f_train = open("train_pcap_length.txt","w")
f_test = open("test_pcap_length.txt","w")
lines = f1.readlines()
for line in lines:
	x = random.randint(1,10)
	if x == 1:
		f_test.write(line)
	else:
		f_train.write(line)
f1.close()
f_train.close()
f_test.close()