import numpy as np
from pesq import pesq
from pesq.cypesq import cypesq_retvals
from pystoi import stoi
import torchaudio
import librosa
import pysepm



transform= torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, win_length=256, power=None)

def mySNR(wave1, wave2):
    de = sum((wave2-wave1)**2)
    nu = sum(wave2**2)

    snr = 10*np.log10((nu/de))

    return snr

def myPESQ(wave1, wave2, fs=16000):
    pesq = cypesq_retvals(fs,wave2,wave1,1)

    return pesq

def myLSD(wave1, wave2):

    
    window = np.hanning(256)
    spec1 = librosa.core.spectrum.stft(wave1, n_fft = 512, hop_length = 128, win_length = 256, window=window)
    spec2 = librosa.core.spectrum.stft(wave2, n_fft = 512, hop_length = 128, win_length = 256, window=window)
    spec1 = np.log(abs(spec1))
    spec2 = np.log(abs(spec2))

    lsd = np.mean([((np.mean((spec_-spec_pred_)**2))**(1/2)) for spec_,spec_pred_ in zip(spec2.T,spec1.T)])
    #print(lsd)

    return lsd

def myLSD_v2(wave1, wave2):


    window = np.hanning(256)
    spec1 = librosa.core.spectrum.stft(wave1, n_fft = 512, hop_length = 128, win_length = 256, window=window)
    spec2 = librosa.core.spectrum.stft(wave2, n_fft = 512, hop_length = 128, win_length = 256, window=window)
    spec1 = np.log10(abs(spec1)**2)
    spec2 = np.log10(abs(spec2)**2)

    original_target_squared = (spec1 - spec2)**2
    lsd = np.mean(np.sqrt(np.mean(original_target_squared, axis=0)))

    return lsd


def mySTOI(wave1, wave2, fs=16000):
    stoi_ = stoi(wave2, wave1, fs, extended=False)

    return stoi_

def myComposite(wave1, wave2, fs=16000):
    csig,cbak,covl = pysepm.composite(wave2, wave1, fs)

    return csig, cbak, covl

