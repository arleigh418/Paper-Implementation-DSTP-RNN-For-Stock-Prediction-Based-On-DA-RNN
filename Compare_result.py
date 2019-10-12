from ops import *
from torch.autograd import Variable

import torch
from torch import cuda
# torch.cuda.is_available()
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import DSTP_rnn,Encoder,Decoder
from model_Darnn import DA_rnn,Encoder,Decoder

def count_values(truth,pred):
    count_avg = 0
    assert len(truth)==len(pred)
    for x in range(len(truth)):
        count_avg+=abs(truth[x]-pred[x])
    return count_avg/len(truth)


if  __name__ == '__main__':
    epoch_list = [3000,5000]
    dstp_result = []
    darnn_result = []
    X, y= read_data("2324.TW.csv", debug=False)
    for i in epoch_list:

        model_darnn = DA_rnn(X, y, 10, 128, 128, 128, 0.01, i)
        model_darnn.load_state_dict(torch.load('darnn_model_{}.pkl'.format(i)))
        
        y_pred_darnn = model_darnn.test()
        y_test_real_darnn = list(y[model_darnn.train_timesteps:])
        darnn_result.append(count_values(y_test_real_darnn,y_pred_darnn))
        


        model_DSTP = DSTP_rnn(X, y, 10 , 128, 128, 128, 0.01, i)
        model_DSTP.load_state_dict(torch.load('dstprnn_model_{}.pkl'.format(i)))
        
        y_pred_dstp = model_DSTP.test()
        y_test_real_dstp = list(y[model_DSTP.train_timesteps:])
        dstp_result.append(count_values(y_test_real_dstp,y_pred_dstp))
        

    

epoch_list_plt = ['3000','5000']

fig3 = plt.figure()
plt.title(u'DSTP-RNN V.S. DA-RNN')
plt.xlabel(u'EPOCH')
plt.ylabel(u"Average Error of Predict Result")
plt.plot(epoch_list_plt,dstp_result,'g-',label = 'DSTP-RNN')
plt.plot(epoch_list_plt,darnn_result, 'r-',label = 'DA-RNN')
# plt.plot(store_method_cos)
plt.legend(loc='best')
plt.savefig("final.png")
plt.close(fig3)
print('Finished comparing')


        
