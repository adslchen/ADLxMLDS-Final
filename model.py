from complexnn import ComplexConv1D, ComplexDense, ComplexBN


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class DCSNet(nn.Module):
    
    def __init__(self):
        super(DCSNet, self).__init__()
        self.main = nn.Sequential(
            # state size. 2 x 1025
            ComplexConv1D(1, 16, 7, 2, 3, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 32 x 513
            ComplexConv1D(16, 32, 5, 2, 2, bias=False),

            #nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 64 x 257
            ComplexConv1D(32, 32, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 64 x 129
            ComplexConv1D(32, 64, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 128 x 65
            ComplexConv1D(64, 64, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. 128 x 33
        )
        self.dense1 = ComplexDense(2112,1025, init_criterion='he')
        self.dense2 = ComplexDense(2112,1025, init_criterion='he')

    def forward(self, input):
        output = self.main(input)
        output = output.view(input.size(0),-1)
        output1 = self.dense1(output)
        output2 = self.dense2(output)
        return output1.view(input.size(0), -1),output2.view(input.size(0),-1)

class DCSNet_bn(nn.Module):
    
    def __init__(self):
        super(DCSNet_bn, self).__init__()
        self.main = nn.Sequential(
            # state size. 2 x 1025
            ComplexConv1D(1, 16, 7, 2, 3, bias=False),
            ComplexBN(16),
            #nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(),
            
            # state size. 32 x 513
            ComplexConv1D(16, 32, 5, 2, 2, bias=False),
            ComplexBN(32),
            #nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(),
            
            # state size. 64 x 257
            ComplexConv1D(32, 32, 3, 2, 1, bias=False),
            ComplexBN(32),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(),
            
            # state size. 64 x 129
            ComplexConv1D(32, 64, 3, 2, 1, bias=False),
            ComplexBN(64),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(), 
            
            # state size. 128 x 65
            ComplexConv1D(64, 64, 3, 2, 1, bias=False),
            ComplexBN(64),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU()
            # state size. 128 x 33
        )
        self.dense1 = ComplexDense(2112,1025, init_criterion='he')
        self.dense2 = ComplexDense(2112,1025, init_criterion='he')

    def forward(self, input):
        output = self.main(input)
        output = output.view(input.size(0),-1)
        output1 = self.dense1(output)
        output2 = self.dense2(output)
        return output1.view(input.size(0), -1),output2.view(input.size(0),-1)



class DCSNet_new(nn.Module):
    
    def __init__(self):
        super(DCSNet_new, self).__init__()
        self.main = nn.Sequential(
            # state size. 2 x 1025
            ComplexConv1D(1, 16, 7, 2, 3, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            nn.ReLU(0.2, inplace=True),
            
            # state size. 32 x 513
            ComplexConv1D(16, 32, 5, 2, 2, bias=False),

            #nn.BatchNorm2d(ngf * 2),
            nn.akyReLU(0.2, inplace=True),
            
            # state size. 64 x 257
            ComplexConv1D(32, 32, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 64 x 129
            ComplexConv1D(32, 64, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. 128 x 65
            ComplexConv1D(64, 64, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. 128 x 33
        )
        self.dense1 = ComplexDense(2112,1025, init_criterion='he')
        self.dense2 = ComplexDense(2112,1025, init_criterion='he')

    def forward(self, input):
        output = self.main(input)
        print("output_size",output_size)
        output = output.view(input.size(0),-1)
        output1 = self.dense1(output)
        output2 = self.dense2(output)
        return output1.view(input.size(0), -1),output2.view(input.size(0),-1)

