import os, errno
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision.utils as vutils
#torch.manual_seed(1)

import numpy as np
from numpy import linalg as LA
from tqdm import tqdm, trange
import random
from random import shuffle
import time
import math
from itertools import groupby
import argparse

import sys
import model
import dataset
# from . import model
from utils import cal_param

cudnn.benchmark=True
def union_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def valid(args,model,ValidX,ValidY):
    print('---Starting Validation---')
    model.eval()
    criterion = nn.MSELoss().cuda()
    if args.kld == True:
        criterion2 = nn.KLDivLoss()

    ValidX = Variable(torch.from_numpy(ValidX).float()).cuda()
    target1 = Variable(torch.from_numpy(ValidY[:,:,:1025].reshape(ValidY.shape[0],-1)).float()).cuda()
    target2 = Variable(torch.from_numpy(ValidY[:,:,1025:].reshape(ValidY.shape[0],-1)).float()).cuda()
    
    valid_size = ValidX.size(0)
    batch_size = 512
    n_iteration = math.ceil(valid_size/batch_size)
    
    total_loss = 0
    for i in range(n_iteration):
        b_X = ValidX[i*batch_size:(i+1)*batch_size]
        b_t1 = target1[i*batch_size:(i+1)*batch_size]
        b_t2 = target2[i*batch_size:(i+1)*batch_size]
        output1,output2 = model(b_X)
        loss = criterion(output1,b_t1)+criterion(output2,b_t2)
        if args.kld == True:
            loss += (criterion2(output1,b_t1)+criterion2(output2,b_t2))


        total_loss += loss.data[0]
    print('Validation loss: %.3f'%(total_loss/n_iteration))
    model.train()
    return total_loss


def load_latest_model(dirname):
    model_list = glob.glob(os.path.join(dirname, "*.pth"))
    max_idx = 0
    max_ep = 0
    
    # (fp.split("/")[-1].split(".")[0][5:])






def train(args):
    net = model.DCSNet_bn()
    if args.kld == True:
        criterion2 = nn.KLDivLoss()

    net.cuda()
    model_dir = os.path.join('models',args.model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.MSELoss().cuda()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()


    Dataset = dataset.Mir1k(args.local_data, complex_=args.complex_, fourier=args.fourier,
                    stft=args.stft)

    Xvalid, Yvalid = Dataset.eval_set('valid')
    print ("valid shape:",Xvalid.shape, Yvalid.shape)
    Xtest, Ytest = Dataset.eval_set('test')
    print ("test shape:",Xtest.shape, Ytest.shape)


    scheduler = lr_scheduler.StepLR(optimizer, 1, 0.9)
    it = Dataset.train_iterator()
    loss_log = [] 
    for epoch in range(args.epochs):
        t = trange(1000)
        total_loss = 0
        index = 0
        for data, i in zip(it, t):
            index += 1
            net.zero_grad()
            features_out = data[0]
            output = data[1]
            # print("output:", output.shape)
            # print("features out:", features_out.shape)
            # sample_size = features_out.shape[0] // 10
            # indice = np.random.permutation(features_out.shape[0])
            # features_out = features_out[indice[:sample_size]]
            # output = output[indice[:sample_size]]

            output1_var = np.squeeze(output[:,:,:1025]).reshape(output.shape[0],-1)
            output2_var = np.squeeze(output[:,:,1025:]).reshape(output.shape[0],-1)

            feat_var = Variable(torch.from_numpy(features_out).float()).cuda()
            output1_var = Variable(torch.from_numpy(output1_var).float()).cuda()
            output2_var = Variable(torch.from_numpy(output2_var).float()).cuda()

            # print("labels",output1_var.size()) 

            my_pred1, my_pred2 = net(feat_var)
            # print("prediction", my_pred1)
            # print("prediction", my_pred2)


            loss1 = criterion(my_pred1, output1_var)
            loss2 = criterion(my_pred2, output2_var)
            loss = loss1 + loss2
            cos_loss_value =0 
            if args.kld == True:
                loss += (criterion2(my_pred1,output1_var)+criterion2(my_pred2,output2_var))
            if args.cos == True:
                cos_loss = torch.mean(torch.abs(cos(my_pred1,my_pred2)))
                cos_loss_value = cos_loss.data[0]
                loss = loss + 100*cos_loss


            total_loss += loss.data[0] 
            loss_log.append(loss.data[0])
            loss.backward()
            optimizer.step()
            t.set_description('loss {} cos loss {}'.format(loss.data[0],cos_loss_value) )
        scheduler.step()
        print("Epoch {} Total Loss : {}".format(epoch,total_loss/1000))
        valid_loss = valid(args,net,Xvalid,Yvalid)
        if epoch % args.save_per_ep == 0:
            torch.save(net.state_dict(),os.path.join(model_dir,'epoch{}.pth'.format(epoch)))
        if epoch % args.save_per_ep == 0:
            np.save(os.path.join(model_dir,'loss_log.npy'),loss_log)
            


    total_loss = 0
    start_time = time.time()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-data',type=str, default='mir-1k/train_data')
    parser.add_argument('--complex', dest='complex_', action='store_true',
                        default=True)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--fourier', action='store_true', default=False)
    parser.add_argument('--stft', action='store_true', default=True)
    parser.add_argument('--save_per_ep', default=5, type=int)
    parser.add_argument('--model_dir',type=str, default='models')
    parser.add_argument('--kld', action='store_true', default=False)
    parser.add_argument('--cos', action='store_true', default=False)

    args = parser.parse_args()
    train(args)

    

