"""
Reference  https://github.com/Zhenye-Na/DA-RNN
"""
# -*- coding: utf-8 -*-

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

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)


class Encoder(nn.Module):
    

    def __init__(self, T ,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):

        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T
       
        
        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.encoder_num_hidden)

        self.encoder_lstm2 = nn.LSTM(
            input_size=self.input_size, hidden_size=self.encoder_num_hidden)

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1, out_features=1, bias=True) #1033
        
        # W_s[h_{t-1} ; s_{t-1}] + U_s[x^k ; y^k]
        self.encoder_attn2 = nn.Linear(
            in_features=2 * self.encoder_num_hidden + 2*self.T - 2, out_features=1, bias=True)

        
    def forward(self, X ,y_prev):
        """forward.

        Args:
            X

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())
        
        X_tilde2 = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())

        X_encoded2 = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # hidden, cell: initial states with dimention hidden_size
        #X --> 233 9 363
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        hs_n = self._init_states(X)
        ss_n = self._init_states(X)
        # y_prev = y_prev.view()
       
        y_prev = y_prev.view(len(X) , self.T-1 ,1)
        
        # print(h_n.size())  # 1 233 512
        # print(s_n.size())
   
        
        for t in range(self.T - 1):
            #Phase one attention
            # batch_size * input_size * (2*hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2), #233 363 1033
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)
            
           
       
            # test = x.view(-1, self.encoder_num_hidden * 2 + self.T - 1)
            # print(test.size()) #84579 1033
            
            x = self.encoder_attn( #84579 1  
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1)) 
         
            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size))# 233x363
            
            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :]) #233x363
            # print(x_tilde.size())
                
            


            # encoder LSTM 
            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            
            

            #Phase two attention

            x2 = torch.cat((hs_n.repeat(self.input_size, 1, 1).permute(1, 0, 2), #233 363 1042
                           ss_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X_tilde.permute(0, 2, 1),
                           y_prev.repeat(1, 1, self.input_size).permute(0, 2, 1)), dim=2)
            
            x2 = self.encoder_attn2( 
                x2.view(-1, self.encoder_num_hidden * 2 + 2*self.T - 2)) 
            
            alpha2 = F.softmax(x2.view(-1, self.input_size))# 233x363
            
            x_tilde2 = torch.mul(alpha2, x_tilde)
            

            self.encoder_lstm2.flatten_parameters()
            _, final_state2 = self.encoder_lstm2(
                x_tilde2.unsqueeze(0), (hs_n, ss_n))
            hs_n = final_state2[0]
            ss_n = final_state2[1]
            # print(x_tilde2.size())
            X_tilde2[:, t, :] = x_tilde2
            X_encoded2[:, t, :] = hs_n
            
            

        return X_tilde2 , X_encoded2

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = Variable(X.data.new(
            1, X.size(0), self.encoder_num_hidden).zero_())
        return initial_states


class Decoder(nn.Module):


    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
                                        nn.Tanh(),
                                        nn.Linear(encoder_num_hidden, 1))
        self.lstm_layer = nn.LSTM(
            input_size=1, hidden_size=decoder_num_hidden)
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final_price = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)
        self.fc_final_trend = nn.Linear(decoder_num_hidden + encoder_num_hidden, 3)
        self.fc_final_trade = nn.Linear(decoder_num_hidden + encoder_num_hidden, 3)

        self.fc.weight.data.normal_()

    def forward(self, X_encoed, y_prev):
        """forward."""
        d_n = self._init_states(X_encoed)
        c_n = self._init_states(X_encoed)

        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoed), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1))
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoed)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))
                # 1 * batch_size * decoder_num_hidden
                d_n = final_states[0]
                # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]
        # Eqn. 22: final output
        final_temp_y = torch.cat((d_n[0], context), dim=1)
        y_pred_price = self.fc_final_price(final_temp_y)
        y_pred_trend = F.softmax(self.fc_final_trend(final_temp_y))
        y_pred_trade = F.softmax(self.fc_final_trade(final_temp_y))
        return y_pred_price, y_pred_trend, y_pred_trade

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = X.data.new(
            1, X.size(0), self.decoder_num_hidden).zero_()
        return initial_states



class DSTP_rnn(nn.Module):
    

    def __init__(self, X, y, trade, trend, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel=False):
       
        super(DSTP_rnn, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.X = X
        self.y = y
        self.trade = trade
        self.trend = trend

        self.Encoder = Encoder(input_size=X.shape[1],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T)
        self.Encoder = self.Encoder.cuda()
        self.Decoder = self.Decoder.cuda()
        # Loss function
        self.criterion_price = nn.MSELoss()
        self.criterion_trend = nn.CrossEntropyLoss()
        self.criterion_trade = nn.CrossEntropyLoss()
    


        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)
        
        # Training set
        self.train_timesteps = int(self.X[:490].shape[0]) #改
        # print(self.train_timesteps)
        self.input_size = self.X.shape[1]

    def train(self):
        best_dev_acc = 0.0 #trade
        
        """training process."""
        iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)
        # trade = self.X[:, 64]
        # trend = self.X[:, 65]
        # print(trade)
        # print(trend)
        n_iter = 0
        # print(trade)
        # print(trend)
        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while (idx < self.train_timesteps):
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                
                # print(indices)
                y_gt = self.y[indices + self.T-1]
                trade_gt = self.trade[indices + self.T-1]
                trend_gt = self.trend[indices + self.T-1]
           
                # trade_gt = self.trade[indices + self.T]
                # format x into 3D tensor
                for bs in range(len(indices)):
                    
                    x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]:(indices[bs] + self.T - 1)]
                   
                loss,acc_trade,acc_trend = self.train_forward(x, y_prev, y_gt,trend_gt,trade_gt)
                
                #loss_trade = self.train_forward2(x, y_prev,trade_gt)
                
                    # no_up = 0
                # else:
                #     no_up += 1
                #     if no_up >= 1000:
                #         exit()
                self.iter_losses[epoch * iter_per_epoch + idx // self.batch_size] = loss
                

                idx += self.batch_size
                n_iter += 1

                if n_iter % 30000 == 0 and n_iter != 0: #3000 500 10000 1500
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])
                # self.epoch_losses2[epoch] = np.mean(self.iter_losses2[range(epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])
                # self.epoch_losses3[epoch] = np.mean(self.iter_losses3[range(epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            if epoch % 10 == 0:
                print ("Epochs: ", epoch, " Iterations: ", n_iter, " Loss: ", self.epoch_losses[epoch] ,'Trade_acc',acc_trade , 'Trend_acc' , acc_trend)
                

    


    def train_forward(self, X, y_prev, y_gt,trend_gt,trade_gt):
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()),Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).cuda())) #cuda
        y_pred_price, y_pred_trend, y_pred_trade = self.Decoder(input_encoded, Variable(
            torch.from_numpy(y_prev).type(torch.FloatTensor)).cuda())#cuda
        # print(y_pred_trade)
        y_true_price = torch.from_numpy(
            y_gt).type(torch.FloatTensor)
        
        y_true_price =y_true_price.view(-1, 1).cuda() #cuda
        
        
        y_true_trend = torch.from_numpy(
            trend_gt).type(torch.LongTensor).cuda() #cuda
        
        # y_true_trend =y_true_trend.view(-1, 3)
        y_true_trade = torch.from_numpy(
            trade_gt).type(torch.LongTensor).cuda() #cuda
        # y_true_trade =y_true_trade.view(-1, 3)
        
        y_trade_dev = torch.max(y_pred_trade,1)[1]
        y_trend_dev =torch.max(y_pred_trend,1)[1] 
        # print(len(y_pred_trade))
        # print(len(y_trade_dev))
        acc_trade = get_accuracy(y_true_trade,y_trade_dev)
        acc_trend = get_accuracy(y_true_trend,y_trend_dev)

        # print(y_pred_trend)
        # print(y_true_trend)
        loss1 = self.criterion_price(y_pred_price, y_true_price)
        loss2 = self.criterion_trend(y_pred_trend, y_true_trend)
        loss3 = self.criterion_trade(y_pred_trade, y_true_trade)
        # loss_total = loss+loss2+loss3
        # loss = loss1+loss2+loss3 
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        # loss = loss1+loss2+loss3
        
        # print('loss_trend:',loss2)
        # print('loss_trade:',loss3)
        # print('acc_trade:',acc_trade,'acc_trend:',acc_trend)

        
        
       
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        

        return loss1.item(),acc_trade,acc_trend
        # ,loss_trade.item()

    

    def val(self):
        """validation."""
        pass


    def test(self, on_train=False):
        """test."""

        if on_train:
            y_pred_price = np.zeros(self.train_timesteps - self.T + 1)
            y_pred_trend = np.zeros(self.train_timesteps - self.T + 1)
            y_pred_trade = np.zeros(self.train_timesteps - self.T + 1)
        else:
            y_pred_price = np.zeros(self.X.shape[0] - self.train_timesteps)
            y_pred_trend = np.zeros(self.X.shape[0] - self.train_timesteps)
            y_pred_trade = np.zeros(self.X.shape[0] - self.train_timesteps)


        i = 0
        while i < len(y_pred_price):
            batch_idx = np.array(range(len(y_pred_price)))[i : (i + self.batch_size)]
            # print(batch_idx)
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                else:

                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_timesteps - self.T,  batch_idx[j]+ self.train_timesteps - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
            _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()),Variable(y_history).cuda()) #cuda

            

            # y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            y_pred_price, y_pred_trend, y_pred_trade = self.Decoder(input_encoded, y_history) #cuda
            
            y_pred_price = y_pred_price[i:(i + self.batch_size)]
            y_pred_price = y_pred_price.cpu().detach().numpy()[:, 0]
            
            y_pred_trend =  y_pred_trend[i:(i + self.batch_size)]
            
        
            y_pred_trade = y_pred_trade[i:(i + self.batch_size)]
            
            
            
            i += self.batch_size
        return y_pred_price , torch.max(y_pred_trade,1)[1] , torch.max(y_pred_trend,1)[1]


X, y,trade,trend= read_data("2324.TW_no_art.csv", debug=False)

model = DSTP_rnn(X, y, trade, trend, 10 , 128, 128, 128, 0.001, 30000)
# model = torch.load('GOODONE.pkl',map_location='cpu')
# model.load_state_dict(torch.load('best_model_acc83808380.model',map_location='cpu')) #this is for model.state_dict()
y_train = model.train()

y_pred,y_pred_trade,y_pred_trend = model.test()

print(y_pred)
print(y_pred_trade)
print(y_pred_trend)



pd_store= pd.DataFrame(columns = ['預測價格','買賣','趨勢'])
pd_store['預測價格'] = y_pred
pd_store['買賣'] = y_pred_trade.cpu()
pd_store['趨勢'] = y_pred_trend.cpu()



pd_store.to_excel('dstp_stage1.xlsx')
        