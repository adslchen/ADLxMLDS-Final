import torch
import numpy as np
from mir_eval.separation import bss_eval_sources

def cal_param(model,name):
    pa = 0
    for param in model.parameters():
        size = param.size()
        p = 1
        for i in size:
            p *= i
        pa += p
    print('# of %s model\'s parameters :'%(name),pa)




def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    # crop
    length_list = []   
    for i in range(len(mixed_wav)):
        crop_length = pred_src1_wav[i].shape[0]
        length_list.append(crop_length)
        mixed_wav[i] = mixed_wav[i][:crop_length]
        src1_wav[i] = src1_wav[i][:crop_length]
        src2_wav[i] = src2_wav[i][:crop_length]
        # pred_src1_wav[i] = pred_src1_wav[i][:100000]
        # pred_src2_wav[i] = pred_src2_wav[i][:100000]

    # for i in range(len(mixed_wav)):
        # print("mixed:", mixed_wav[i].shape)
        # print("src1", src1_wav[i].shape)
        # print("src2", src2_wav[i].shape)
        # print("pred src1", pred_src1_wav[i].shape)
        # print("pred src2", pred_src2_wav[i].shape)
    length = len(mixed_wav)
    gnsdr = np.zeros(2)
    gsir  = np.zeros(2)
    gsar = np.zeros(2)
    total_len = 0

    for i in range(length):
        sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav[i], src2_wav[i]]),
                                            np.array([pred_src1_wav[i], pred_src2_wav[i]]), False)

        sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav[i], src1_wav[i]]),
                                              np.array([mixed_wav[i], mixed_wav[i]]), False)
        print("sdr {}, sir {}, sar {}".format(sdr,sir,sar))
        nsdr = sdr - sdr_mixed
        gnsdr += length_list[i] * nsdr
        gsir += length_list[i] * sir
        gsar += length_list[i] * sar
        total_len += length_list[i]
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    return gnsdr, gsir, gsar


