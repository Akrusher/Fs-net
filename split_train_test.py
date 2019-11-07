import random
f1 = open("NIMS.arff","r")
f_train = open("NIMS_train.arff","w")
f_test = open("NIMS_test.arff","w")
lines = f1.readlines()
for i in range(28,len(lines)):
	x = random.randint(0,20)
	line = lines[i]
	if x == 0:
		f_test.write(line)
	else:
		f_train.write(line)
f1.close()
f_train.close()
f_test.close()