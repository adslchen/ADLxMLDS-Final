# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk

import itertools
import numpy as np

from itertools import chain
from scipy import fft
from scipy.signal import stft

# Robert edit
import os
from os import listdir
from os.path import isfile, join
import preprocess
import random

FS = 44100            # samples/second
DEFAULT_WINDOW_SIZE = 2048    # fourier window size, default: 2048
OUTPUT_SIZE = 2025               # number of distinct notes, default: 128
STRIDE = 1024          # samples between windows, default: 512
WPS = FS / float(512)   # windows/second

# Robert edit
STFT_WIN = 2048

class Mir1k(object):
    def __init__(self, filename, in_memory=True, window_size=4096, # window_size = 4096, output_size = 84 
                 output_size=2050, feature_size=1024, sample_freq=11000,
                 complex_=False, fourier=False, stft=True, fast_load=False,
                 rng=None, seed=123):
        if not in_memory:
            raise NotImplementedError
        self.filename = filename

        self.window_size = window_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.sample_freq = sample_freq
        self.complex_ = complex_
        self.fourier = fourier
        self.stft = stft
        self.fast_load = fast_load

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(seed)

        # Robert edit
        self.file_list = np.array([join(filename, f) for f in listdir(filename) if isfile(join(filename, f))])
        self.train_inds, self.valid_inds, self.test_inds = self.splits(len(self.file_list))

        self._train_data = {}
        self._valid_data = {}
        self._test_data = {}
        self._loaded = False

        self._eval_sets = {}
    """
    def splits(self):
        with open(self.filename, 'rb') as f:
            # This should be fast
            all_inds = np.load(f).keys()
        test_inds = ['2303', '2382', '1819']
        valid_inds = ['2131', '2384', '1792',
                      '2514', '2567', '1876']
        train_inds = [ind for ind in all_inds
                      if ind not in test_inds and ind not in test_inds]
        return train_inds, valid_inds, test_inds
    """
    # Robert edit
    def splits(self, file_num):
        all_inds = np.arange(int(file_num/10*9))
        #valid_inds = np.random.choice(file_num, int(file_num/10), replace=False)
        valid_inds = np.arange(int(file_num/10*9), file_num)
        test_inds = valid_inds[int(len(valid_inds)/3*2):]
        valid_inds = valid_inds[:int(len(valid_inds)/3*2)]
        train_inds = [ind for ind in all_inds
                      if ind not in test_inds and ind not in test_inds]
        return train_inds, valid_inds, test_inds
 
    @classmethod
    def note_to_class(cls, note):
        return note - 21

    @property
    def train_data(self):
        if self._train_data == {}:
            self.load()
        return self._train_data

    @property
    def valid_data(self):
        if self._valid_data == {}:
            self.load()
        return self._valid_data

    @property
    def test_data(self):
        if self._test_data == {}:
            self.load()
        return self._test_data

    # Robert edit
    """
    def load(self, filename=None, reload=False):
        if filename is None:
            filename = self.filename
        if self._loaded and not reload:
            return
        
        with open(filename, 'rb') as f:
            train_inds, valid_inds, test_inds = self.splits()
            data_file = np.load(f)
            if self.fast_load:
                train_inds = train_inds[:6]
                train_data = {}
                for ind in chain(train_inds, valid_inds, test_inds):
                    train_data[ind] = data_file[ind]
            else:
                train_data = dict(data_file)

            # test set
            test_data = {}
            for ind in test_inds:
                if ind in train_data:
                    test_data[ind] = train_data.pop(ind)

            # valid set
            valid_data = {}
            for ind in valid_inds:
                valid_data[ind] = train_data.pop(ind)

            self._train_data = train_data
            self._valid_data = valid_data
            self._test_data = test_data
    """
    def load(self,batchsize, _inds):
        for _id in range(0,len(self.train_inds),batchsize):
            wavname = self.file_list[_inds[_id:_id+batchsize]]
            npyname = ['data/data_%d_%d%s' %(STFT_WIN, STRIDE,
                        wav[wav.rfind('/'):wav.rfind('.')]+'.npy') for wav in wavname]
            loadwav_fn = [f for f in npyname if not isfile(f)]
            
            if len(loadwav_fn) != 0:
                train_map = preprocess.load_single_wav(wavname)
                train_data = train_map["mixed"]
                train_tar1 = train_map["src1"]
                train_tar2 = train_map["src2"]
                
                if not os.path.exists('./data/data_%d_%d' %(STFT_WIN, STRIDE)):
                    os.makedirs('./data/data_%d_%d' %(STFT_WIN, STRIDE))
                for b in range(batchsize):
                    np.save(npyname[b],{'mixed':train_map["mixed"][b],'src1':train_map["src1"][b],
                            'src2':train_map["src2"][b]})
                
            else:
                train_map = [np.load(f).item() for f in npyname]
                train_data = [_map["mixed"] for _map in train_map]
                train_tar1 = [_map["src1"] for _map in train_map]
                train_tar2 = [_map["src2"] for _map in train_map]
             
            return train_data,train_tar1,train_tar2

    # Robert edit
    def construct_eval_set(self, data, tar1, tar2,step=128):
        n_files = len(data)
        pos_per_file = 100
        features = np.empty([n_files * pos_per_file, self.window_size])
        output_1 = np.zeros([n_files * pos_per_file, self.window_size])
        output_2 = np.zeros([n_files * pos_per_file, self.window_size])

        features_ind = 0
        labels_ind = 1

        for i, ind in enumerate(data):
            #print(ind)
            #audio = data[i]

            for j in range(pos_per_file):
                #if j % 100 == 0:
                #    print(j)
                # start from one second to give us some wiggle room for larger
                # segments
                index = self.sample_freq + j * step
                features[pos_per_file * i + j] = data[i][index:index + self.window_size]
                output_1[pos_per_file * i + j] = tar1[i][index:index + self.window_size]
                output_2[pos_per_file * i + j] = tar2[i][index:index + self.window_size]
                """
                # label stuff that's on in the center of the window
                s = int((index + self.window_size / 2))
                for label in data[ind][labels_ind][s]:
                    note = label.data[1]
                    outputs[pos_per_file * i + j, self.note_to_class(note)] = 1
                """
        return features, output_1, output_2

    @property
    def feature_dim(self):
        dummy_features = np.zeros((1, self.window_size))
        dummy_output = np.zeros((1, self.window_size))
        dummy_features, _ = self.aggregate_raw_batch(
            dummy_features, dummy_output, dummy_output)
        return dummy_features.shape[1:]

    # Robert edit
    def aggregate_raw_batch(self, features, output_1, output_2):
        """Aggregate batch.

        All post processing goes here.

        Parameters:
        -----------
        features : 3D float tensor
            Input tensor
        output : 2D integer tensor
            Output classes

        """
        channels = 2 if self.complex_ else 1
        if self.fourier:
            if self.complex_:
                data = fft(features, axis=1)
                features_out[:, :, 0] = np.real(data[:, :, 0])
                features_out[:, :, 1] = np.imag(data[:, :, 0])
            else:
                data = np.abs(fft(features, axis=1))
                features_out = data
        elif self.stft:
            """
            features_out = np.zeros(
                [features.shape[0], self.window_size, channels])
            # default: nperseg=120, noverlap=60
            _, _, data = stft(features, nperseg=DEFAULT_WINDOW_SIZE, noverlap=STRIDE, axis=1)
            length = data.shape[1]
            n_feats = data.shape[3]
            """
            data, tar1, tar2 = preprocess.make_train(features, output_1, output_2) 
            if self.complex_:
                features_out = np.zeros(
                    [data.shape[0], data.shape[1], channels])
                features_out[:, :, 0] = np.real(data)
                features_out[:, :, 1] = np.imag(data)
                
                output_1 = np.zeros(
                    [data.shape[0], data.shape[1], channels])
                output_1[:, :, 0] = np.real(tar1)
                output_1[:, :, 1] = np.imag(tar1)
                
                output_2 = np.zeros(
                    [data.shape[0], data.shape[1], channels])
                output_2[:, :, 0] = np.real(tar2)
                output_2[:, :, 1] = np.imag(tar2)

                output = np.concatenate((output_1,output_2),1)

                features_out = np.transpose(features_out, (0,2,1))
                output = np.transpose(output, (0,2,1))

                return (features_out, output)
            else:
                features_out = np.zeros(
                    [data.shape[0], data.shape[1], channels])
                features_out[:, :, 0] = np.abs(data)
                
                features_phase = np.zeros(
                    [data.shape[0], data.shape[1], channels])
                features_phase[:, :, 0] = np.angle(data)
                
                tar = np.concatenate((tar1,tar2), 1)
                output = np.zeros(
                    [tar.shape[0], tar.shape[1], channels])
                output[:,:,0] = np.abs(tar)

                output_phase = np.zeros(
                    [tar.shape[0], tar.shape[1], channels])
                output_phase[:,:,0] = np.angle(tar)
                
                features_out = np.transpose(features_out, (0,2,1))
                output = np.transpose(output, (0,2,1))
                features_phase = np.transpose(features_phase, (0,2,1))
                output_phase = np.transpose(output_phase, (0,2,1))

                return (features_out, output, features_phase, output_phase)
        else:
            features_out = features
        return (features_out, output)
    
    # Robert edit
    def train_iterator(self):
        train_data, train_tar1, train_tar2 = self.load(len(self.train_inds),self.train_inds)
        # print ("train song:",len(self.train_inds))
        # print ("valid song:",len(self.valid_inds))
        # print ("test song:",len(self.test_inds))

        while True:
            # sample file from train data
            sample_size = len(train_data) // 10
            indice = list(range(len(train_data)))
            select_indice = random.sample(indice,k=sample_size)
            sample_data = [ train_data[idx] for idx in select_indice ]
            sample_tar1 = [ train_tar1[idx] for idx in select_indice ] 
            sample_tar2 = [ train_tar2[idx] for idx in select_indice ] 

            features = np.zeros([len(sample_data), self.window_size])
            output_1 = np.zeros([len(sample_data), self.window_size])
            output_2 = np.zeros([len(sample_data), self.window_size])
            for j, ind in enumerate(sample_data):
                s = self.rng.randint(
                    self.window_size // 2,
                    len(sample_data[j]) - self.window_size // 2)
                data = sample_data[j][s - self.window_size // 2:
                                        s + self.window_size // 2]
                tar1 = sample_tar1[j][s - self.window_size // 2:
                                        s + self.window_size // 2]
                tar2 = sample_tar2[j][s - self.window_size // 2:
                                        s + self.window_size // 2]
                features[j, :] = data
                output_1[j, :] = tar1
                output_2[j, :] = tar2

                """
                for label in self.train_data[ind][1][s]:
                    note = label.data[1]
                    output[j, self.note_to_class(note)] = 1
                """
            yield self.aggregate_raw_batch(features, output_1, output_2)

    def eval_set(self, set_name):
        if not self._eval_sets:
            for name in ['valid', 'test']:
                _inds = self.valid_inds if name == 'valid' else self.test_inds
                data, tar1, tar2 = self.load(len(_inds),_inds)
                data, tar1, tar2 = self.construct_eval_set(data, tar1, tar2)
                x = self.aggregate_raw_batch(data, tar1, tar2)
                self._eval_sets[name] = x
        return self._eval_sets[set_name]
