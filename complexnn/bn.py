import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import sys
from operator import mul
import math

class ComplexBatchNormalization(nn.Module):
    def __init__(self,
                 channel_dim,
                 epsilon = 1e-5,
                 momentum = 0.9
                 ):
        super(ComplexBatchNormalization, self).__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.channel_dim = channel_dim
        self.initialize_parameters()
    def initialize_parameters(self):
        # moving average initialization
        #self.Vrr_moving = Variable(torch.ones(self.channel_dim, 1, 1) * 1/math.sqrt(2), requires_grad=False)

        self.Vrr_moving = Variable(torch.cuda.FloatTensor([1/math.sqrt(2)] * (self.channel_dim)), requires_grad=False)
        self.Vri_moving = Variable(torch.cuda.FloatTensor([0] * (self.channel_dim)), requires_grad=False)
        self.Vii_moving = Variable(torch.cuda.FloatTensor([1/math.sqrt(2)] * (self.channel_dim)), requires_grad=False)
        self.moving_mean = Variable(torch.cuda.FloatTensor([0] * (self.channel_dim * 2)), requires_grad=False)
        
        self.Vrr_moving  = self.Vrr_moving.view(self.Vri_moving.size(0), 1, 1) 
        self.Vri_moving  = self.Vri_moving.view(self.Vri_moving.size(0), 1, 1) 
        self.Vii_moving  = self.Vii_moving.view(self.Vii_moving.size(0), 1, 1) 
        self.moving_mean = self.moving_mean.view(self.moving_mean.size(0), 1, 1)
        
        #self.beta_real_moving = nn.Parameter(torch.cuda.FloatTensor([0]))
        #self.beta_imag_moving  = nn.Parameter(torch.cuda.FloatTensor([0]))
        # scaling param initialization
        self.gamma_rr = nn.Parameter(torch.cuda.FloatTensor([1/math.sqrt(2)] * self.channel_dim))
        self.gamma_ri = nn.Parameter(torch.cuda.FloatTensor([0] * self.channel_dim))
        self.gamma_ii = nn.Parameter(torch.cuda.FloatTensor([1/math.sqrt(2)] * self.channel_dim))
        self.beta = nn.Parameter(torch.cuda.FloatTensor([0] * (self.channel_dim * 2)))
        #self.beta_imag = nn.Parameter(torch.cuda.FloatTensor([0]))
        
    def moving_mean_update(self, moving_mean, mu, momentum):
        return (moving_mean * momentum + mu * (1. - momentum))
        
    def complex_standardization(self, input_centred, Vrr, Vii, Vri):
        input_shape = input_centred.size() # 32 x 20 x 5 x 5
        ndim = len(input_shape) # 4
        channel_dim = input_shape[0] // 2 # channel size / 2
        
        tau = Vrr + Vii
        delta = (Vrr * Vii) - Vri**2
        
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        
        inverse_st = 1.0 / (s*t)
        Wrr = ((Vii + s) * inverse_st).view(channel_dim, 1, 1) # 16 x 1 x 1
        Wii = ((Vrr + s) * inverse_st).view(channel_dim, 1, 1)
        Wri = (-Vri * inverse_st).view(channel_dim, 1 ,1)
        
        W_cat_real = torch.cat([Wrr, Wii], dim = 0) # 32 x 1 x 1
        W_cat_imag = torch.cat([Wri, Wri], dim = 0) # 32 x 1 x 1
 
        if ndim == 3:
            centred_real = input_centred[:channel_dim, :, :]
            centred_imag = input_centred[channel_dim:, :, :]
        else:
            sys.exit('Sorry! Have not handled the case that input_dim != 3')
        
        rolled_input = torch.cat([centred_imag, centred_real], dim = 0)
        output = W_cat_real * input_centred + W_cat_imag * rolled_input
        #print(output.size()) # 30 x 20 x 5
        return output
        
    def ComplexBN(self, input_centred, Vrr, Vii, Vri, 
                       beta, 
                       gamma_rr, gamma_ri, gamma_ii):
        input_shape = input_centred.size() # 32 x 20 x 5 x 5
        ndim = len(input_shape) # 4
        channel_dim = input_shape[0] // 2 # channel size / 2

        standardized_output = self.complex_standardization(input_centred, Vrr, Vii, Vri)

        broadcast_gamma_rr = gamma_rr.view(channel_dim, 1, 1)
        broadcast_gamma_ri = gamma_ri.view(channel_dim, 1, 1)
        broadcast_gamma_ii = gamma_ii.view(channel_dim, 1, 1)
        
        gamma_cat_real = torch.cat([broadcast_gamma_rr, broadcast_gamma_ii], dim = 0)
        gamma_cat_imag = torch.cat([broadcast_gamma_ri, broadcast_gamma_ri], dim = 0)
        
        if ndim == 3:
            centred_real = standardized_output[:channel_dim, :, :]
            centred_imag = standardized_output[channel_dim:, :, :]
        else:
            sys.exit('Sorry! Have not handled the case that input_dim != 3')
            
        rolled_standardized_output = torch.cat([centred_imag, centred_real], dim = 0)
        
        broadcast_beta = beta.view(channel_dim * 2, 1, 1)
        returned = gamma_cat_real * standardized_output + gamma_cat_imag * rolled_standardized_output + broadcast_beta # 32 x 20 x 5
        returned = returned.permute(1, 0, 2).contiguous()
        return returned
    
    def forward(self, x):
        input_shape = x.size() # 20 x 32 x 5
        ndim = len(input_shape) # 3
        channel_dim = input_shape[1] // 2 # channel size / 2
        
        
        x_permute = x.permute(1, 0, 2).contiguous() # 32 x 20 x 5
        # channel is at first dim now

        if(self.training == False):
            # print('Complex batch normalization is testing...')
            inference_centred = x_permute - self.moving_mean
            returned = self.ComplexBN(inference_centred, self.Vrr_moving, self.Vii_moving,
                             self.Vri_moving, self.beta, self.gamma_rr, self.gamma_ri,
                             self.gamma_ii) # 20 x 32 x 5
            return returned
        else:
            # print('Complex batch normalization is training...')
            x_reshape = x_permute.view(channel_dim * 2, -1) # 32 x 100
            mu = torch.mean(x_reshape, dim = 1).view(channel_dim * 2, 1, 1) # 32 x 1 x 1
            
            input_centred = x_permute - mu # 32 x 20 x 5
            centred_squared = input_centred ** 2  # 32 x 20 x 5

            
            if ndim == 3:
                centred_squared_real = centred_squared[:channel_dim, :, :] # 16 x 20 x 5 
                centred_squared_imag = centred_squared[channel_dim:, :, :]
                centred_real = input_centred[:channel_dim, :, :]
                centred_imag = input_centred[channel_dim:, :, :]
            else:
                sys.exit('Sorry! Have not handled the case that input_dim != 3')
            
    
            Vrr = (torch.mean(centred_squared_real.view(channel_dim, -1), dim = 1) + self.epsilon).view(channel_dim, 1, 1) # 16
            Vii = (torch.mean(centred_squared_imag.view(channel_dim, -1), dim = 1) + self.epsilon).view(channel_dim, 1, 1) # 16
            Vri = (torch.mean((centred_real * centred_imag).view(channel_dim, -1), dim = 1)).view(channel_dim, 1, 1) # 16
            
            self.moving_mean = self.moving_mean_update(self.moving_mean, mu, self.momentum)
            self.Vrr_moving = self.moving_mean_update(self.Vrr_moving, Vrr, self.momentum)
            self.Vri_moving = self.moving_mean_update(self.Vri_moving, Vri, self.momentum)
            self.Vii_moving = self.moving_mean_update(self.Vii_moving, Vii, self.momentum)
        
            
            input_bn = self.ComplexBN(input_centred, Vrr, Vii, Vri, 
                                self.beta, 
                                self.gamma_rr, self.gamma_ri, self.gamma_ii) # 32 x 20 x 5
            
            
            x_permute = input_bn.permute(1, 0, 2).contiguous() # 32 x 20 x 5
            x_reshape = x_permute.view(channel_dim * 2, -1) # 32 x 100
            mu = torch.mean(x_reshape, dim = 1)
            
            big_gamma = (Vrr + Vii).view(channel_dim)
            C = (Vrr - Vii).view(channel_dim)

            # print('E(x_bar): {}'.format(torch.mean(mu).data.cpu()[0]))
            # print('Cov(x_bar): {}'.format(torch.mean(big_gamma).data.cpu()[0]))
            # print('Pseudo_Cov(x_bar): {}'.format(torch.mean(C).data.cpu()[0]))

            return input_bn
   
        #print(centred_squared)

if __name__ == '__main__':

    input_real = torch.ones(20,16,5).uniform_(0,1)
    input_imag  = torch.ones(20,16,5).uniform_(0,1)
    input_complex = torch.cat([input_real, input_imag], dim = 1)
    cbn = ComplexBatchNormalization(channel_dim = int(input_complex.size(1) / 2)).cuda()
    cbn.initialize_parameters()
    #cbn.eval()
    input = Variable(input_complex).cuda()
    #print(input.size())
    output = cbn(input)
    #print(output.size())
    """a = Variable(torch.ones(10,3).uniform_(0,1))
    print(a.size())
    b = torch.mean(a, dim = -1).unsqueeze(1)
    print(b.size())
    c = a-b
    print(c)
    print(torch.mean(c, dim=1))
    """
"""def __init__(self,
                 axis=-1,
                 momentum=0.9,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer='sqrt_init',
                 gamma_off_initializer='zeros',p
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='sqrt_init',
                 moving_covariance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 beta_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 **kwargs
                 ):"""
