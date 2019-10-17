from numpy import *

def TrainNetwork( img, label_batch_list ,Data):

    hid_act1 =   maximum(0, matmul(img, Data.W1) + Data.b1)
    hid_act2 =   maximum(0, matmul(hid_act1, Data.W2) + Data.b2)
    scores =   maximum(0,matmul(hid_act2, Data.W3) + Data.b3)

    scores_e = exp(scores)
    scores_e_sum = sum(scores_e, axis=1, keepdims=True)

    probs = scores_e / scores_e_sum
    loss_list_tmp= sum(scores_e * label_batch_list, axis=1, keepdims=True)/scores_e_sum
    loss_list = - log(loss_list_tmp)

    loss =  mean(loss_list, axis=0)
    Data.loss.append(loss)

    out_delta=probs-label_batch_list
    out_delta /= Data.BATCHSIZE

    hid_delta2 =   dot(out_delta, Data.W3.T)
    hid_delta2[hid_act2 <= 0] = 0

    hid_delta1 =   dot(hid_delta2, Data.W2.T)
    hid_delta1[hid_act1 <= 0] = 0

    Data.W3 = Data.W3 - Data.stepsize *(dot(hid_act2.T,out_delta)+Data.reg_factor*Data.W3)
    Data.W2 = Data.W2 - Data.stepsize * (dot(hid_act1.T,hid_delta2 )+Data.reg_factor*Data.W2)
    Data.W1 = Data.W1 - Data.stepsize * (dot(img.T, hid_delta1)+Data.reg_factor*Data.W1)

    Data.b3 = Data.b3 - Data.stepsize * sum( out_delta, axis = 0, keepdims= True )
    Data.b2 = Data.b2 - Data.stepsize * sum( hid_delta2, axis = 0, keepdims= True )
    Data.b1 = Data.b1 - Data.stepsize * sum( hid_delta1, axis = 0, keepdims= True )
    return

def testNetwork(img,Data):

    hid_act1 =   maximum(0, matmul(img, Data.W1) + Data.b1)
    hid_act2 =   maximum(0,matmul(hid_act1, Data.W2) + Data.b2)
    scores =   maximum(0,matmul(hid_act2, Data.W3) + Data.b3)
    return scores