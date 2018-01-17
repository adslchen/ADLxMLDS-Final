import torch
import numpy as np 
from numpy.random import RandomState
from torch.autograd import Variable
import math, random
from torch.nn.init import _calculate_fan_in_and_fan_out
import sys
import time
import functools


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
    return newfunc


def ComplexIndependentFilters(tensor, init_criterion, seed):
    if isinstance(tensor, Variable):
        ComplexIndependentFilters(tensor.data, init_criterion, seed)
        return tensor
    filter_type = None
    if len(tensor.size()) == 3:
        filter_type = 'Conv1d'
        num_rows = int(tensor.size()[0]/2) * tensor.size()[1]
        num_cols = tensor.size()[2]
        kernel_size = (tensor.size()[2],)
    elif len(tensor.size()) == 4:
        filter_type = 'Conv2d'
        num_rows = int(tensor.size()[0]/2) * tensor.size()[1]
        num_cols = tensor.size()[2] * tensor.size()[3]
        kernel_size = (tensor.size()[2], tensor.size()[3])
    else:
        sys.exit('The convolution type not support.')
    flat_shape = (int(num_rows), int(num_cols))
    rng = RandomState(seed)
    r = rng.uniform(size=flat_shape)
    i = rng.uniform(size=flat_shape)
    z = r + 1j * i
    u, _, v = np.linalg.svd(z)
    unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
    real_unitary = unitary_z.real
    imag_unitary = unitary_z.imag

    indep_real = np.reshape(real_unitary, (num_rows,) + kernel_size)
    indep_imag = np.reshape(imag_unitary, (num_rows,) + kernel_size)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if init_criterion == 'glorot':
        desired_var = 1./(fan_in + fan_out)
    elif init_criterion == 'he':
        desired_var = 1./fan_in
    else:
        raise ValueError('invalid init critierion',init_criterion)

    multip_real = np.sqrt(desired_var / np.var(indep_real))
    multip_imag = np.sqrt(desired_var / np.var(indep_imag))
    scaled_real = multip_real * indep_real
    scaled_imag = multip_imag * indep_imag

    kernel_shape = (int(tensor.size()[0]/2), tensor.size()[1]) + kernel_size
    weight_real = np.reshape(scaled_real, kernel_shape)
    weight_imag = np.reshape(scaled_imag, kernel_shape)

    weight = np.concatenate([weight_real,weight_imag], axis=0)
    temp_weight = torch.from_numpy(weight).float()
    tensor.copy_(temp_weight)
    del temp_weight
    return tensor


def ComplexInit(tensor, init_criterion, seed):
    if isinstance(tensor, Variable):
        ComplexInit(tensor.data, init_criterion, seed)
        return tensor
    filter_type = None
    if len(tensor.size()) == 3:
        filter_type = 'Conv1d'
        num_rows = int(tensor.size()[0]/2) * tensor.size()[1]
        num_cols = tensor.size()[2]
        kernel_size = (tensor.size()[2],)
    elif len(tensor.size()) == 4:
        filter_type = 'Conv2d'
        num_rows = int(tensor.size()[0]/2) * tensor.size()[1]
        num_cols = tensor.size()[2] * tensor.size()[3]
        kernel_size = (tensor.size()[2], tensor.size()[3])
    else:
        sys.exit('The convolution type not support.')
    kernel_shape = (int(tensor.size()[0]/2), tensor.size()[1]) + kernel_size
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if init_criterion == 'glorot':
        desired_var = 1./(fan_in + fan_out)
    elif init_criterion == 'he':
        desired_var = 1./fan_in
    else:
        raise ValueError('invalid init critierion',init_criterion)
    
    rng = RandomState(seed)
    modulus = rng.rayleigh(scale=desired_var, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    weight = np.concatenate([weight_real,weight_imag], axis=0)
    temp_weight = torch.from_numpy(weight).float()
    tensor.copy_(temp_weight)
    del temp_weight
    return tensor

if __name__ == "__main__":
    conv1 = torch.nn.Conv2d(3,16,4)
    kargs = {'seed':100,'init_criterion':'glorot'}
    start_time = time.time()
    w_init = ComplexInit
    weights = w_init(conv1.weight,**kargs)

    print(weights)
    end_time = time.time()
    print(end_time - start_time)
    
