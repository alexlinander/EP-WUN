from torch.utils.data import Dataset
import torch
import torchaudio
import numpy as np
from glob import glob

class BWENoiseDataset(Dataset):
    def __init__(self, type='train',language='CHN', noise=None, snr=20, scale=1):
        self.type = type
        self.language = language
        self.noise = noise
        self.snr = snr
        self.scale = scale
        self.n = glob('/opt/software/kaldi/egs/librispeech/s5_noise/esun_8k_noise_threshold_45k/noise_segment_16k/*.wav')
        if language == 'CHN_v2':
            # if type == "train":
            #     self.x = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_trainset/*_x/*.wav')
            #     self.y = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_trainset/*_y/*.wav')
            # elif type == "valid":
            #     self.x = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_validset/*_x/*.wav')
            #     self.y = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_validset/*_y/*.wav')
            # elif type == "test":
            #     self.x = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_testset/*_x/*.wav')
            #     self.y = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_testset/*_y/*.wav')
            if type == "train":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_trainset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_trainset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_trainset/*_y/*.wav')
            elif type == "valid":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_validset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_validset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_validset/*_y/*.wav')
            elif type == "test":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_testset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_testset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_testset/*_y/*.wav')

        elif language == 'ENG':
            if type == "train":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_trainset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_trainset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_trainset/*_y/*.wav')
            elif type == "valid":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_validset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_validset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_validset/*_y/*.wav')
            elif type == "test":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_testset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_testset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_testset/*_y/*.wav')

    def __getitem__(self,index):
            
        if self.language == 'CHN':
            if self.noise==None:
                x_path = self.x[index//2]
                y_path = self.y[index//2]
                x,_ = torchaudio.load(x_path, normalize = True)
                y,_ = torchaudio.load(y_path, normalize = True)
                if index%2 == 0:
                    noise_index = torch.randint(0,60875,(1,))
                    n_path = self.n[noise_index]
                    n,_ = torchaudio.load(n_path, normalize = True)
                    x = self.snr_noise_mix(x ,n, self.snr)

            else:
                x_path = self.x[index]
                y_path = self.y[index]
                x,_ = torchaudio.load(x_path, normalize = True)
                y,_ = torchaudio.load(y_path, normalize = True)
                if self.noise == 'noisy':
                    noise_index = torch.randint(0,60875,(1,))
                    n_path = self.n[noise_index]
                    n,_ = torchaudio.load(n_path, normalize = True)
                    x = self.snr_noise_mix(x ,n, self.snr)

            return index, x, y

        elif self.language == 'ENG' or 'CHN_v2':
            if self.noise == None:
                if index%2 == 0:
                    x_path = self.x[index//2]
                    label = 1
                else:
                    x_path = self.x_noisy[index//2]
                    label = 0
                y_path = self.y[index//2]
                x,_ = torchaudio.load(x_path, normalize = True)
                y,_ = torchaudio.load(y_path, normalize = True)

            elif self.noise == 'clean':
                x_path = self.x[index//2]
                y_path = self.y[index//2]
                x,_ = torchaudio.load(x_path, normalize = True)
                y,_ = torchaudio.load(y_path, normalize = True)
                label = 1
            elif self.noise == 'noisy':
                x_path = self.x_noisy[index//2]
                y_path = self.y[index//2]
                x,_ = torchaudio.load(x_path, normalize = True)
                y,_ = torchaudio.load(y_path, normalize = True)
                label = 0
            # print(x_path)
            # print(y_path)


            return index, x, y, label

    def __len__(self):
        length = len(self.x)
        length = np.int32(np.floor(length*self.scale))
        
        if self.noise == None or self.type == 'train' :
            return length*2
        else:
            return length

    def snr_noise_mix(self, speech ,noise, snr):
        snr = 10**(snr/10.0)
        speech_power = torch.mean(torch.pow(speech,2))
        noise_power = torch.mean(torch.pow(noise,2))
        noise_update = noise / torch.sqrt(snr * noise_power/speech_power)
        noise_power = torch.mean(torch.pow(noise_update,2))
        return noise_update + speech

class BWEDataset(Dataset):
    def __init__(self, type='train', noise=None):
        self.type = type
        self.noise = noise
        
        if type == "train":
            self.x_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_trainset/*_x/*.wav')
            self.y_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_trainset/*_y/*.wav')
            self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-trainset/*_x/*.wav')
            self.y_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-trainset/*_y/*.wav')
        elif type == "valid":
            self.x_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_validset/*_x/*.wav')
            self.y_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_validset/*_y/*.wav')
            self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-validset/*_x/*.wav')
            self.y_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-validset/*_y/*.wav')
        elif type == "test":
            self.x_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_testset/*_x/*.wav')
            self.y_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_testset/*_y/*.wav')
            self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-testset/*_x/*.wav')
            self.y_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-testset/*_y/*.wav')

    def __getitem__(self,index):
        
        if self.noise == None:
            if index%2 == 0:
                x_path = self.x_clean[index//2]
                y_path = self.y_clean[index//2]
            else:
                x_path = self.x_noisy[index//2]
                y_path = self.y_noisy[index//2]
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)

        elif self.noise == 'clean':
            x_path = self.x_clean[index]
            y_path = self.y_clean[index]
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)
        elif self.noise == 'noisy':
            x_path = self.x_noisy[index]
            y_path = self.y_noisy[index]
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)

        return index, x, y

    def __len__(self):
        length = len(self.x_clean)
        if self.noise == None:
            return length
        else:
            return length


class SQCDataset(Dataset):
    def __init__(self, type='train', noise=None):
        self.type = type
        self.noise = noise
    
        if type == "train":
            self.x_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_trainset/*_x/*.wav')
            self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-trainset/*_x/*.wav')
        elif type == "valid":
            self.x_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_validset/*_x/*.wav')
            self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-validset/*_x/*.wav')
        elif type == "test":
            self.x_clean = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_testset/*_x/*.wav')
            self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy-testset/*_x/*.wav')

    def __getitem__(self,index):
        if self.noise == None:
            if index%2 == 0:
                x_path = self.x_clean[index//2]
                x,_ = torchaudio.load(x_path, normalize = True)
                y = [1,0]
            else:
                x_path = self.x_noisy[index//2]
                x,_ = torchaudio.load(x_path, normalize = True)
                y = [0,1]
        else:
            if self.noise == 'clean':
                x_path = self.x_clean[index]
                x,_ = torchaudio.load(x_path, normalize = True)
                y = [1,0]
            elif self.noise == 'noisy':
                x_path = self.x_noisy[index]
                x,_ = torchaudio.load(x_path, normalize = True)
                y = [0,1]

        return index, x, np.float32(y)

    def __len__(self):
        length = len(self.x_clean)
        if self.noise == None:
            return length*2
        else:
            return length

