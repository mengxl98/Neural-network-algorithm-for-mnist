import numpy as np
import mnist_download as load
import network2 as net
import matplotlib.pyplot as plt
import datetime
import csv

class Data:
    W1 = 0.01 * np.random.randn(28 * 28, 100)
    b1 = 0.01 * np.random.randn(1, 100)

    W2 = 0.01 * np.random.randn(100, 20)
    b2 = 0.01 * np.random.randn(1, 20)

    W3 = 0.01 * np.random.randn(20, 10)
    b3 = 0.01 * np.random.randn(1, 10)

    reg_factor = 0.0001
    stepsize = 0.035
    loss=[]


trainim,trainlal=load.load_mnist('','train')
testim, testlal=load.load_mnist('','t10k')
trainim=trainim/256.0
testim=testim/256.0

K=1500
H=5
BATCHSIZE=400
train_num = len(trainim)
out_num =10
t_label = np.zeros((train_num, out_num))

for i in range(train_num):
    t_label[i, trainlal[i]] = 1

acclist = []

for i in range(K):
    sizet = (i * BATCHSIZE) % 60000
    sizee = ((i + 1) * BATCHSIZE) % 60000
    if sizet >= sizee:
        continue
    img_list = trainim[sizet:sizee, :]
    label_list = t_label[sizet:sizee, :]
    net.TrainNetwork(img_list,label_list,Data)

result=net.testNetwork(testim,Data)
acc=0
for t in range(len(result)):
    if np.argmax(result[t,:]) == testlal[t]:
        acc = acc + 1
acclist.append(acc / 10000.0)

print acclist
x1=range(len(Data.loss))
plt.figure(figsize=(5, 3))
plt.plot(x1,Data.loss,label='loss fuction',linewidth=1)
plt.show()