import librosa 
import numpy as np
import os, glob
import pickle

# 0: background , 1: vocal
# wavfiles should be a list
def load_single_wav(wavfiles):
    wavs = [ librosa.load(wf, mono=False)[0] for wf in wavfiles]
    mixed = [ librosa.to_mono(wav) for wav in wavs]
    src1 = [ wav[0] for wav in wavs ]
    src2 = [ wav[1] for wav in wavs ]
    return {'mixed':np.array(mixed), 'src1':np.array(src1), 'src2':np.array(src2)} 

def real_load_single_wav(filename):
    wav = librosa.load(filename, mono=False)[0]
    mixed = np.array(librosa.to_mono(wav))
    src1 = np.array(wav[0])
    src2 = np.array(wav[1])
    return mixed, src1, src2


def load_wav(dirname):
    print("loading wav file ...")
    wavfiles = glob.glob(os.path.join(dirname,"*"))
    wavs = [ librosa.load(wf, mono=False)[0] for wf in wavfiles]
    mixed = [ librosa.to_mono(wav) for wav in wavs]
    src1 = [ wav[0] for wav in wavs ]
    src2 = [ wav[1] for wav in wavs ]
    return mixed, src1, src2

# Robert edit 
def make_train(mixed, src1, src2):
    #mixed, src1, src2 = load_wav(dirname)
    # print("to spe cctrogram ...")
    mixed = to_spectrogram(mixed)
    src1  = to_spectrogram(src1)
    src2  = to_spectrogram(src2)
    """
    mixed_spectrogram = to_spectrogram(mixed)
    src1_spectrogram = to_spectrogram(src1)
    src2_spectrogram = to_spectrogram(src2)
    mixed = []
    src1 = []
    src2 = []
    for m, s1, s2 in zip(mixed_spectrogram, src1_spectrogram, src2_spectrogram):
        mixed.append(m)
        src1.append(s1)
        src2.append(s2)
    """
    mixed_arr = np.transpose(np.hstack(tuple(mixed)))
    #mixed_arr = np.transpose(np.array(list(get_real_complex(mixed_arr))),(2,1,0))
    src1_arr = np.transpose(np.hstack(tuple(src1)))
    #src1_arr = np.transpose(np.array(list(get_real_complex(src1_arr))),(2,1,0))
    src2_arr = np.transpose(np.hstack(tuple(src2)))
    #src2_arr = np.transpose(np.array(list(get_real_complex(src2_arr))),(2,1,0))

    return mixed_arr, src1_arr, src2_arr



def split_complex(features):
    features_out = np.zeros((features.shape[0],features.shape[1], 2))
    features_out[:,:,0] = np.real(features)
    features_out[:,:,1] = np.imag(features)
    return features_out


# Batch considered
def to_spectrogram(wav, len_frame=2048, len_hop=1024):
    return list(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav))


# Batch considered
def to_wav(mag, phase, len_hop=1024):
    stft_maxrix = get_stft_matrix(mag, phase)
    return map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix)


# Batch considered
def to_wav_from_spec(stft_maxrix, len_hop=1024):
    return list(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix))


# Batch considered
def to_wav_mag_only(mag, init_phase, len_frame=2048, len_hop=1024, num_iters=50):
     
    #wav_list = griffin_lim(np.transpose(mag),len_frame,len_hop, num_iters=num_iters,phase_angle=np.transpose(init_phase))
    #return wav_list
    return list(map(lambda m,p: griffin_lim(m, len_frame, len_hop, num_iters=num_iters,phase_angle=p),mag, init_phase))

# Batch considered
def get_magnitude(stft_matrixes):
    return np.abs(stft_matrixes)


# Batch considered
def get_phase(stft_maxtrixes):
    return np.angle(stft_maxtrixes)


# Batch considered
def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)

# Lo add
def get_real_complex(matrix):
    return matrix.real,((matrix-matrix.real)*-1j).real

def griffin_lim(mag, len_frame, len_hop, num_iters, phase_angle=None, length=None):
    assert(num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = get_stft_matrix(mag, phase_angle)
    for i in range(num_iters):
        wav = librosa.istft(spec, hop_length=len_hop)
        if i != num_iters - 1:
            spec = librosa.stft(wav, n_fft=len_frame, win_length=len_frame, hop_length=len_hop)
            _, phase = librosa.magphase(spec)
            phase_angle = np.angle(phase)
            spec = get_stft_matrix(mag, phase_angle)
    return wav

if __name__ == "__main__":
    train = make_train('mir-1k/small_Wavfile')
    #with open('data/data.p','wb') as f:
    #    pickle.dump(train, f)
    np.save("data/small_data_2048_512.npy",train)
