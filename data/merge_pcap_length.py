import os
f1 = open("iqiyi_pcap_length.txt","r")
f2 = open("taobao_pcap_length.txt","r")
f3 = open("weibo_pcap_length.txt","r")
f4 = open("weixin_pcap_length.txt","r")
f = open("pcap_length.txt","w")
lines = f1.readlines()
for line in lines:
	file_name = "iqiyi_" + line.strip().split()[0]
	pkt_len_list = line.strip().split()[1:]
	f.write(file_name + " ")
	for i, pkt_len in enumerate(pkt_len_list):
		if i == 512:
			break
		f.write(pkt_len + " ")
	f.write("\n")
lines = f2.readlines()
for line in lines:
	file_name = "taobao_" + line.strip().split()[0]
	pkt_len_list = line.strip().split()[1:]
	f.write(file_name + " ")
	for i, pkt_len in enumerate(pkt_len_list):
		if i == 512:
			break
		f.write(pkt_len + " ")
	f.write("\n")
lines = f3.readlines()
for line in lines:
	file_name = "weibo_" + line.strip().split()[0]
	pkt_len_list = line.strip().split()[1:]
	f.write(file_name + " ")
	for i, pkt_len in enumerate(pkt_len_list):
		if i == 512:
			break
		f.write(pkt_len + " ")
	f.write("\n")
lines = f4.readlines()
for line in lines:
	file_name = "weixin_" + line.strip().split()[0]
	pkt_len_list = line.strip().split()[1:]
	f.write(file_name + " ")
	for i, pkt_len in enumerate(pkt_len_list):
		if i == 512:
			break
		f.write(pkt_len + " ")
	f.write("\n")
f.close()