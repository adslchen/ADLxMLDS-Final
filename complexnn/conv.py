import torch
import torch.nn as nn
from torch.autograd import Variable
from .init import ComplexIndependentFilters,ComplexInit
import sys


class ComplexConv1D(nn.Module):
    def __init__(self,in_channel,out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 init_criterion='he',
                 kernel_initializer='complex_independent',
                 bias_initializer='zeros',
                 kernel_constraint=None,
                 bias_constraint=None,
                 activation=None
                 ): 
        super(ComplexConv1D,self).__init__()
        if activation != None:
            sys.exit('Activation have not been create!')
        self.bias = bias
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.Real_kernel = nn.Conv1d(self.in_channel,self.out_channel,self.kernel_size,stride,padding,dilation,
                                     groups,self.bias)
        self.Img_kernel = nn.Conv1d(self.in_channel,self.out_channel,self.kernel_size,stride,padding,dilation,
                                     groups,self.bias)
        self.init_criterion = init_criterion
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.initialize_parameters() 
    def initialize_parameters(self):
        r_k = self.Real_kernel.weight.data
        i_k = self.Img_kernel.weight.data
        con_kernel = torch.cat([r_k,i_k],dim=0)
        if torch.initial_seed() > 1e2: seed = None
        else: seed = torch.initial_seed()
        ## kernel_initializer
        if self.kernel_initializer in {'complex','complex_independent'}:
            kernel_init = {'complex':ComplexInit,
                           'complex_independent':ComplexIndependentFilters}[self.kernel_initializer]
            if self.init_criterion not in {'glorot','he'}:
                sys.exit('No Pre-define init-criterion: %s'%(self.init_criterion))
        else:
            kernel_init = None
        if kernel_init != None:
            kernel_init(con_kernel,self.init_criterion,seed)  
        ## bias_initializer
        if self.bias and self.bias_initializer=='zeros':
            nn.init.constant(self.Real_kernel.bias,0)
            nn.init.constant(self.Img_kernel.bias,0)
        
        ## Final weight initialization
        self.Real_kernel.weight.data = con_kernel[:self.out_channel]
        self.Img_kernel.weight.data = con_kernel[self.out_channel:]
    def forward(self,input):
        # input:(batch_size,2*in_channel,L_in)
        # output: (batch_size, 2*out_channel,L_out)
        assert(input.dim()==3)
        assert(input.size(1)%2==0)
        assert(input.size(1)//2==self.in_channel)

        real_input = input[:,:self.in_channel]
        img_input = input[:,self.in_channel:]
        
        Real_real = self.Real_kernel(real_input)
        Img_img = self.Img_kernel(img_input)
        Real_img = self.Real_kernel(img_input)
        Img_real = self.Img_kernel(real_input)

        Real_part = Real_real - Img_img
        Img_part = Real_img + Img_real

        output = torch.cat([Real_part,Img_part],dim=1)

        return output
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channel=' + str(self.in_channel) \
            + ', out_channel=' + str(self.out_channel) \
            + ', kernel_size=' + str(self.kernel_size)\
            + ', bias=' + str(self.bias is not None) + ')'

if __name__ == '__main__':

    model = ComplexConv1D(3,6,3,padding=1).cuda()
    print(model)

    input = torch.Tensor(5,6,25).cuda()
    input = Variable(input)

    output = model(input)

    print(output.size())
        


