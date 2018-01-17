import torch
import torch.nn as nn
from torch.autograd import Variable
import sys


class ComplexDense(nn.Module):
    def __init__(self,in_feature,out_feature,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_constraint=None,
                 bias_constraint=None
                 ):
        super(ComplexDense,self).__init__()
        if activation != None:
            sys.exit('Activation have not been create!')
        self.use_bias = use_bias
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Real_matrix = nn.Linear(self.in_feature,self.out_feature,bias=self.use_bias)
        self.Img_matrix = nn.Linear(self.in_feature,self.out_feature,bias=self.use_bias)
        self.init_criterion = init_criterion
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.initialize_parameters()

    def initialize_parameters(self):
        if self.kernel_initializer == 'complex':
            if self.init_criterion == 'he':
                init_w = nn.init.kaiming_normal
                init_b = nn.init.constant
                kwargs = {'a':0,'mode':'fan_in'}
            elif self.init_criterion == 'glorot':
                init_w = nn.init.xavier_normal
                init_b = nn.init.constant
                kwargs = {'gain':nn.init.calculate_gain('linear')}
            else:
                sys.exit('No pre-define complex initializer')
        else:
            init_w = None
        if init_w != None:
            #for real
            init_w(self.Real_matrix.weight,**kwargs)
            if self.use_bias == True:
                init_b(self.Real_matrix.bias,0)
            #for img
            init_w(self.Img_matrix.weight,**kwargs)
            if self.use_bias == True:
                init_b(self.Img_matrix.bias,0)
        ## Kernel Constraint
        if self.kernel_constraint == 'max_norm':
            norm = torch.sqrt(torch.sum(self.Real_matrix.weight.data**2,dim=1,keepdim=True))
            desired = torch.clamp(norm,0,2)
            self.Real_matrix.weight.data *= (desired/(1e-7+norm))
            norm = torch.sqrt(torch.sum(self.Img_matrix.weight.data**2,dim=1,keepdim=True))
            desired = torch.clamp(norm,0,2)
            self.Img_matrix.weight.data *= (desired/(1e-7+norm))
        elif self.kernel_constraint == None:
            pass
        else:
            sys.exit('No pre-define kernel_constraint')
        ## bias constraint
        if self.bias_constraint == 'max_norm' and self.use_bias:
            norm = torch.sqrt(torch.sum(self.Real_matrix.bias.data**2,dim=0,keepdim=True))
            desired = torch.clamp(norm,0,2)
            self.Real_matrix.bias.data *= (desired/(1e-7+norm))
            norm = torch.sqrt(torch.sum(self.Img_matrix.bias.data**2,dim=0,keepdim=True))
            desired = torch.clamp(norm,0,2)
            self.Img_matrix.bias.data *= (desired/(1e-7+norm))
        elif self.bias_constraint == None:
            pass
        else:
            sys.exit('No pre-define bias_constraint')


    def forward(self,input):
        ## input_size = (batch_size,2*in_feature)
        ## output_size = (batch_size, 2*out_feature)
        assert(input.dim()==2)
        assert(input.size(1)%2==0)
        assert(input.size(1)//2==self.in_feature)
        real_input = input[:, :self.in_feature]
        img_input = input[:, self.in_feature:]

        Real_real = self.Real_matrix(real_input)
        Img_img = self.Img_matrix(img_input)
        Real_img = self.Real_matrix(img_input)
        Img_real = self.Img_matrix(real_input)

        Real_part = Real_real - Img_img
        Img_part = Real_img + Img_real

        output = torch.cat([Real_part,Img_part],dim=1)

        return output
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_feature) \
            + ', out_features=' + str(self.out_feature) \
            + ', bias=' + str(self.use_bias is not None) + ')'

if __name__ == '__main__':

    cd = ComplexDense(10,20,kernel_constraint='max_norm',bias_constraint='max_norm').cuda()
    print(repr(cd))

    input_real = torch.ones(20,10)
    input_img = torch.ones(20,10)

    input = Variable(torch.cat([input_real,input_img],dim=1)).cuda()

    output = cd(input)

    print(output.size())

        




