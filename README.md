# Paper-Implementation-DSTP-RNN-For-Stock-Prediction-Based-On-DA-RNN 
This Project is based on this paper : DSTP-RNN: a dual-stage two-phase attention-based recurrent neural networks for long-term and multivariate time series prediction(Yeqi Liu, Chuanyang Gong, Ling Yang, Yingyi Chen) https://arxiv.org/ftp/arxiv/papers/1904/1904.07464.pdf

# Introduction
1. This code modify from : https://github.com/Zhenye-Na/DA-RNN (DARNN)
2. I try to implement DSTP-RNN-I model as below picture(This paper introduce two models,please refer this paper's content).We can find that this paper combine yT with first phase attention ouput at second phase attention
![image](https://github.com/arleigh418/Paper-Implementation-DSTP-RNN-For-Stock-Prediction-Based-On-DA-RNN/blob/master/img/DSTP%20PAPER1.png)
(Come from : DSTP-RNN: a dual-stage two-phase attention-based recurrent neural networks for long-term and multivariate time series prediction(Yeqi Liu, Chuanyang Gong, Ling Yang, Yingyi Chen) https://arxiv.org/ftp/arxiv/papers/1904/1904.07464.pdf)

3. add the second phase attention with concat yT as below code.
```
#Phase two attention from DSTP-RNN Paper
x2 = torch.cat((hs_n.repeat(self.input_size, 1, 1).permute(1, 0, 2), #233 363 1042
                ss_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                X.permute(0, 2, 1),
                y_prev.repeat(1, 1, self.input_size).permute(0, 2, 1)), dim=2)      
x2 = self.encoder_attn2( 
     x2.view(-1, self.encoder_num_hidden * 2 + 2*self.T - 2))         
alpha2 = F.softmax(x2.view(-1, self.input_size))   
x_tilde2 = torch.mul(alpha2, x_tilde)
```

4. According to my test,concat X is more better than concat x_tilde that output from first phase attention.
```
#Not better with concat x_tilde.
x2 = torch.cat((hs_n.repeat(self.input_size, 1, 1).permute(1, 0, 2), #233 363 1042
                ss_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                x_tilde.permute(0, 2, 1),
                y_prev.repeat(1, 1, self.input_size).permute(0, 2, 1)), dim=2)      
```

5.I do some very simple test,please refer result,and I will try to improve it.(parameter just like code,you can set the same parameter to test which is better between DA-RNN and DSRP-RNN)
![image](https://github.com/arleigh418/Paper-Implementation-DSTP-RNN-For-Stock-Prediction-Based-On-DA-RNN/blob/master/img/Compare_Darnn_dstprnn.png)

# Final
1. Thanks to all authors of the paper(DSTP-RNN: a dual-stage two-phase attention-based recurrent neural networks for long-term and multivariate time series prediction(Yeqi Liu, Chuanyang Gong, Ling Yang, Yingyi Chen))

2. Thanks to the implementer of DA-RNN code and also this paper authors.


3.If you have any questions , please contact me , cause I'm sure that there must have some misunderstanding to this paper,if you have any suggestions , please kindly let me know.

## I will  keep improving it !
