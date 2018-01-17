import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from os import path
import os
import argparse
import model
import preprocess
import librosa
import glob
from utils import *
import time

def make_test(file_list):

    """
    This function read a list of file path and return the stft tranform 
    as a list

    input: file path list of testing song
    output:
        mixed_batch: list of stft of mixed songs
        src1_batch: list of stft of src1 songs
        src2_batch: list of stft of src2 songs
    """
    
    mixed_batch, src1_batch, src2_batch = [], [], []

    mixed_wavs, src1_wavs, src2_wavs = [], [], []
    for fp in file_list:
        # print("fp",fp)
        mixed, src1 ,src2= preprocess.real_load_single_wav(fp)
        mixed_wavs.append(mixed)
        src1_wavs.append(src1)
        src2_wavs.append(src2)
        # print(mixed)
        # print(src1)
        # print(src2)
        mixed = librosa.stft(mixed, n_fft=2048, hop_length=1024)
        src1 = librosa.stft(src1, n_fft=2048, hop_length=1024)
        src2 = librosa.stft(src2, n_fft=2048, hop_length=1024)

        mixed_batch.append(np.transpose(preprocess.split_complex(mixed),(1,2,0)))
        src1_batch.append(np.transpose(preprocess.split_complex(src1),(1,2,0)))
        src2_batch.append(np.transpose(preprocess.split_complex(src2),(1,2,0)))

    return mixed_batch, src1_batch, src2_batch, mixed_wavs, src1_wavs, src2_wavs
        

def spec_to_wav(features):
    """
    Make the 2 channel in the last dimension to complex
    Input:
        features: a list of 3d array , the last dimension is 2, represent real and imag
    Output:
        wavs_files: a list of 2d array, 
    """
    blob = []
    for feature in features:
        c_feat = feature[:,:,0] + 1j*feature[:,:,1]
        print("c_Feat",c_feat.shape)
        blob.append(c_feat.transpose())
        
    wav_files = preprocess.to_wav_from_spec(blob) 
    print("wavfiles shape", wav_files[0].shape)
    return wav_files



def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # file_list = [join(args.test_dir, f) for f in listdir(args.test_dir) if isfile(join(args.test_dir, f))]
    file_list = glob.glob(args.test_dir+"/*.wav")
    
    print(file_list)
    net = model.DCSNet_bn()
    net.cuda()
    # net.eval()

    # load model
    net.load_state_dict(torch.load(args.model_path))


    mix_list, src1_list, src2_list, mix_wavs, src1_wavs, src2_wavs = make_test(file_list)
    # time.sleep(30)

    print("Testing...")
    pred1_list = []
    pred2_list = []
    for mix, src1, src2 in zip(mix_list, src1_list, src2_list):
        length  = mix.shape[0]
        print("input shape", mix.shape)
        mix = Variable(torch.from_numpy(mix).float()).cuda()
        pred1, pred2 = net(mix)
        pred1 = pred1.data.cpu().numpy()
        pred2 = pred2.data.cpu().numpy()

        
        pred1 = np.transpose(pred1.reshape(length, 2, 1025),(0,2,1))
        pred2 = np.transpose(pred2.reshape(length, 2, 1025),(0,2,1))
        # pred1 = pred1[:,:,0] + pred1[:,:,1] * 1j
        # pred2 = pred2[:,:,0] + pred2[:,:,1] * 1j
        # print("output shape", pred1.shape) 
        # pred1_list.append(pred1.transpose())
        # pred2_list.append(pred2.transpose())
        pred1_list.append(pred1)
        pred2_list.append(pred2)

    # convert the spectogram to time domain
    # pred1_mag = [ preprocess.get_magnitude(mat) for mat in pred1_list]
    # pred1_phase = [preprocess.get_phase(mat) for mat in pred1_list]
    # pred2_mag = [preprocess.get_magnitude(mat) for mat in pred2_list]
    # pred2_phase = [preprocess.get_phase(mat) for mat in pred2_list]
   
    # print("pred1_mag",pred1_mag[0].shape)

    # pred1_wavs = preprocess.to_wav_mag_only(pred1_mag, pred1_phase, len_frame=2048, len_hop=1024, num_iters=50)
    # pred2_wavs = preprocess.to_wav_mag_only(pred2_mag, pred2_phase, len_frame=2048, len_hop=1024, num_iters=50)

    pred1_wavs = spec_to_wav(pred1_list)
    # print("pred1_wav",pred1_wavs)
    pred2_wavs = spec_to_wav(pred2_list)

    # time.sleep(30)
    # calculate GDNSR
    gnsdr, gsir, gsar = bss_eval_global(mix_wavs, src1_wavs, src2_wavs, pred1_wavs, pred2_wavs)
    print("GNSDR music: {} GNSDR vocal: {}, GSIR music: {}, GSIR vocal: {}, GSAR music {}, GSAR vocal: {}".format(gnsdr[0],gnsdr[1], gsir[0], gsir[1], gsar[0], gsar[1]))

    # save the files to output dir
    
    print("Saving the file...")
    for fp, src1, src2 in zip(file_list, pred1_wavs, pred2_wavs):
        file_name = fp.split("/")[-1].split(".")[0]
        output_path = os.path.join(args.output_dir, file_name)
        librosa.output.write_wav(output_path+"_src1.wav", src1, sr=22050 )
        librosa.output.write_wav(output_path+"_src2.wav", src2, sr=22050 )













    




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--test_dir', type=str, default='test_data')
    parser.add_argument('--window_size', type=int, default=2048)
    parser.add_argument('--channel', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()
    main(args)
    
