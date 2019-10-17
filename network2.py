from numpy import *

def sigmoid(inX):
    return 1.0 / (1.0 + exp(-inX))

def softmax(x):
    return maximum(0,x)

def TrainNetwork(img,t_label,Data):

    hid_value1=dot(img,Data.W1)+Data.b1
    hid_act1=sigmoid(hid_value1)
    hid_value2=dot(hid_act1,Data.W2)+Data.b2
    hid_act2=sigmoid(hid_value2)
    out_value=dot(hid_act2,Data.W3)+Data.b3
    out_act=sigmoid(out_value)

    err = t_label - out_act
    Loss=err**2
    Data.loss.append(sum(map(sum, Loss)))
    out_delta=err*out_act*(1-out_act)
    hid_delta2 = hid_act2*(1 - hid_act2) * matmul(out_delta,Data.W3.T)
    hid_delta1 =  hid_act1*(1 - hid_act1) * matmul(hid_delta2,Data.W2.T)

    Data.W3 = Data.W3 + Data.stepsize *(matmul(hid_act2.T,out_delta)+Data.reg_factor*Data.W3)
    Data.W2 = Data.W2 + Data.stepsize * (matmul(hid_act1.T,hid_delta2 )+Data.reg_factor*Data.W2)
    Data.W1 = Data.W1 + Data.stepsize * (matmul(img.T, hid_delta1)+Data.reg_factor*Data.W1)

    Data.b3 = Data.b3 + Data.stepsize * sum( out_delta, axis = 0, keepdims= True )
    Data.b2 = Data.b2 + Data.stepsize * sum( hid_delta2, axis = 0, keepdims= True )
    Data.b1 = Data.b1 + Data.stepsize * sum( hid_delta1, axis = 0, keepdims= True )

    return

def testNetwork(img,Data):
    hid_value1 = matmul(img, Data.W1) + Data.b1
    hid_act1 = sigmoid(hid_value1)
    hid_value2 = matmul(hid_act1, Data.W2) + Data.b2
    hid_act2 = sigmoid(hid_value2)
    out_value = matmul(hid_act2, Data.W3) + Data.b3
    out_act = sigmoid(out_value)
    return out_act