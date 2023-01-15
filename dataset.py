from torch.utils.data import Dataset,DataLoader
import torch
import joblib
import torchaudio
import os
import numpy as np
from glob import glob

class BWEStage2Dataset(Dataset):
    def __init__(self, type='train', language='ENG', noise=None):
        self.type = type
        self.language = language
        self.noise = noise
        if language=='ENG':
            if type == "train":
                self.x_dict = joblib.load('../BWE_dataset/stage2_ln_trainset/stage2_trainset_t_x.pkl')
                self.y_dict = joblib.load('../BWE_dataset/stage2_ln_trainset/stage2_trainset_t_y.pkl')
            elif type == "test":
                self.x_dict = joblib.load('../BWE_dataset/stage2_ln_testset/stage2_testset_x.pkl')
                self.y_dict = joblib.load('../BWE_dataset/stage2_ln_testset/stage2_testset_y.pkl')
            elif type == "valid":
                self.x_dict = joblib.load('../BWE_dataset/stage2_ln_trainset/stage2_trainset_v_x.pkl')
                self.y_dict = joblib.load('../BWE_dataset/stage2_ln_trainset/stage2_trainset_v_y.pkl')
        elif language=='CHN':
            if type == "train":
                self.x_path = ['/mnt/Internal/esun_bwe/CHN_dataset/clean_trainset/clean_trainset_segments_x/',
                                '/mnt/Internal/esun_bwe/CHN_dataset/other_trainset/other_trainset_segments_x/']
                self.y_path = ['/mnt/Internal/esun_bwe/CHN_dataset/clean_trainset/clean_trainset_segments_y/',
                                '/mnt/Internal/esun_bwe/CHN_dataset/other_trainset/other_trainset_segments_y/']
            elif type == "valid":
                if noise=='clean':
                    self.x_path = ['/mnt/Internal/esun_bwe/CHN_dataset/clean_validset/clean_validset_segments_x/']
                    self.y_path = ['/mnt/Internal/esun_bwe/CHN_dataset/clean_validset/clean_validset_segments_y/']
                elif noise=='other':
                    self.x_path = ['/mnt/Internal/esun_bwe/CHN_dataset/other_validset/other_validset_segments_x/']
                    self.y_path = ['/mnt/Internal/esun_bwe/CHN_dataset/other_validset/other_validset_segments_y/']
            elif type == "test":
                if noise=='clean':
                    self.x_path = ['/mnt/Internal/esun_bwe/CHN_dataset/clean_testset/clean_testset_segments_x/']
                    self.y_path = ['/mnt/Internal/esun_bwe/CHN_dataset/clean_testset/clean_testset_segments_y/']
                elif noise=='other':
                    self.x_path = ['/mnt/Internal/esun_bwe/CHN_dataset/other_testset/other_testset_segments_x/']
                    self.y_path = ['/mnt/Internal/esun_bwe/CHN_dataset/other_testset/other_testset_segments_y/']


    def __getitem__(self,index):
        str_ = "segment_" + str(index)
        if self.language == 'ENG':
            return index, self.x_dict[str_], self.y_dict[str_]
        elif self.language == 'CHN':
            if self.type == 'train':
                if index<=555045:
                    str_x = self.x_path[0] + str_ + '.wav'
                    str_y = self.y_path[0] + str_ + '.wav'
                    x = torchaudio.load(str_x, normalize = True)
                    y = torchaudio.load(str_y, normalize = True)
                else:
                    str__ = 'segment_' + str(index-555046)
                    str_x = self.x_path[1] + str__ + '.wav'
                    str_y = self.y_path[1] + str__ + '.wav'
                    x = torchaudio.load(str_x, normalize = True)
                    y = torchaudio.load(str_y, normalize = True)

            else:
                str_x = self.x_path[0] + str_ + '.wav'
                str_y = self.y_path[0] + str_ + '.wav'
                x = torchaudio.load(str_x, normalize = True)
                y = torchaudio.load(str_y, normalize = True)
            return index, x[0], y[0]

    def __len__(self):
        if self.language == 'ENG':
            return len(self.x_dict)
        elif self.language == 'CHN':
            length = 0
            for path in self.x_path:
                length += len(os.listdir(path))
            return length

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
    def __init__(self, type='train', language='CHN', noise=None, snr=20, scale=1):
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
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/*_trainset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Formosa_dataset/*_trainset/*_x/*.wav')
            elif type == "valid":
                if noise == 'clean':
                    self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_validset/*_x/*.wav')
                    self.y = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_validset/*_y/*.wav')
                elif noise == 'noisy':
                    self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_validset/*_x/*.wav')
                    self.y = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_validset/*_y/*.wav')
            elif type == "test":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_testset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_testset/*_y/*.wav')
        elif language == 'ENG':
            if type == "train":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/*_trainset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Valentini_dataset/*_trainset/*_y/*.wav')
            elif type == "valid":
                if noise == 'clean':
                    self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_validset/*_x/*.wav')
                    self.y = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_validset/*_y/*.wav')
                elif noise == 'noisy':
                    self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_validset/*_x/*.wav')
                    self.y = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_validset/*_y/*.wav')
            elif type == "test":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_testset/*_x/*.wav')
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
                    y = self.snr_noise_mix(y ,n, self.snr)

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
                    y = self.snr_noise_mix(y ,n, self.snr)

            return index, x, y

        elif self.language == 'ENG' :
            x_path = self.x[index]
            y_path = self.y[index]

            # print(x_path)
            # print(y_path)
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)

            return index, x, y
        elif self.language == 'CHN_v':
            if self.type == "train":
                if index%2 ==0:
                    x_path = self.x[index//2]
                    y_path = self.y1[index//2]
                else:
                    x_path = self.x_n[index//2]
                    y_path = self.y2[index//2]
            else:
                x_path = self.x[index]
                y_path = self.y[index]

            # print(x_path)
            # print(y_path)
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)

            return index, x, y

    def __len__(self):
        length = len(self.x)
        length = np.int32(np.floor(length*self.scale))
        if self.language == 'CHN' and self.noise == None:
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

class NoiseEncoderDataset(Dataset):
    def __init__(self, type='train', language='CHN', noise=None, snr=20, scale=1):
        self.type = type
        self.language = language
        self.noise = noise
        self.snr = snr
        self.scale = scale
        self.n = glob('/opt/software/kaldi/egs/librispeech/s5_noise/esun_8k_noise_threshold_45k/noise_segment_16k/*.wav')
        if language == 'CHN_v2':
            # if type == "train":
            #     self.x = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_trainset/*_x/*.wav')
            # elif type == "valid":
            #     self.x = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_validset/*_x/*.wav')
            # elif type == "test":
            #     self.x = glob('/mnt/Internal/esun_bwe/CHN_dataset/*_testset/*_x/*.wav')
            if type == "train":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_trainset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_trainset/*_x/*.wav')
            elif type == "valid":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_validset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_validset/*_x/*.wav')
            elif type == "test":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_testset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Formosa_dataset/noisy_testset/*_x/*.wav')
        elif language == 'ENG':
            if type == "train":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_trainset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_trainset/*_x/*.wav')
            elif type == "valid":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_validset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_validset/*_x/*.wav')
            elif type == "test":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_testset/*_x/*.wav')
                self.x_noisy = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_testset/*_x/*.wav')

    def __getitem__(self,index):
        if self.language == 'CHN':
            if self.noise==None:
                x_path = self.x[index//2]
                x,_ = torchaudio.load(x_path, normalize = True)
                if index%2 == 0:
                    noise_index = torch.randint(0,60875,(1,))
                    n_path = self.n[noise_index]
                    n,_ = torchaudio.load(n_path, normalize = True)
                    x = self.snr_noise_mix(x ,n, self.snr)
                    y = [0,1]
                else:
                    y = [1,0]

            else:
                x_path = self.x[index]
                x,_ = torchaudio.load(x_path, normalize = True)
                if self.noise == 'noisy':
                    noise_index = torch.randint(0,60875,(1,))
                    n_path = self.n[noise_index]
                    n,_ = torchaudio.load(n_path, normalize = True)
                    x = self.snr_noise_mix(x ,n, self.snr)
                    y = [0,1]
                else:
                    y = [1,0]
        
        elif self.language == 'ENG' or 'CHN_v2':
            if self.noise == None:
                if index%2 == 0:
                    x_path = self.x[index//2]
                    x,_ = torchaudio.load(x_path, normalize = True)
                    y = [1,0]
                else:
                    x_path = self.x_noisy[index//2]
                    x,_ = torchaudio.load(x_path, normalize = True)
                    y = [0,1]
            else:
                if self.noise == 'clean':
                    x_path = self.x[index]
                    x,_ = torchaudio.load(x_path, normalize = True)
                    y = [1,0]
                elif self.noise == 'noisy':
                    x_path = self.x_noisy[index]
                    x,_ = torchaudio.load(x_path, normalize = True)
                    y = [0,1]
            # print(x_path)
            # print(y)

        return index, x, np.float32(y)

    def __len__(self):
        length = len(self.x)
        length = np.int32(np.floor(length*self.scale))
        if self.noise == None:
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


class MTLMBEDataset(Dataset):
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
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/MTL_MBE/noisy_trainset_8k/*.wav')
                self.y_8k = glob('/mnt/Internal/esun_bwe/Formosa_dataset/MTL_MBE/clean_trainset_8k/*.wav')
                self.y_16k = glob('/mnt/Internal/esun_bwe/Formosa_dataset/MTL_MBE/clean_trainset_16k/*.wav')
            elif type == "valid":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/MTL_MBE/noisy_validset_8k/*.wav')
                self.y_8k = glob('/mnt/Internal/esun_bwe/Formosa_dataset/MTL_MBE/clean_validset_8k/*.wav')
                self.y_16k = glob('/mnt/Internal/esun_bwe/Formosa_dataset/MTL_MBE/clean_validset_16k/*.wav')
            elif type == "test":
                self.x = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_testset/*_x/*.wav')
                self.y = glob('/mnt/Internal/esun_bwe/Formosa_dataset/clean_testset/*_y/*.wav')

        elif language == 'ENG':
            if type == "train":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/MTL_MBE/noisy_trainset_8k/*.wav')
                self.y_8k = glob('/mnt/Internal/esun_bwe/Valentini_dataset/MTL_MBE/clean_trainset_8k/*.wav')
                self.y_16k = glob('/mnt/Internal/esun_bwe/Valentini_dataset/MTL_MBE/clean_trainset_16k/*.wav')
            elif type == "valid":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/MTL_MBE/noisy_validset_8k/*.wav')
                self.y_8k = glob('/mnt/Internal/esun_bwe/Valentini_dataset/MTL_MBE/clean_validset_8k/*.wav')
                self.y_16k = glob('/mnt/Internal/esun_bwe/Valentini_dataset/MTL_MBE/clean_validset_16k/*.wav')
            elif type == "test":
                self.x = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_testset/*_x/*.wav')
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
            x_path = self.x[index]
            y_8kpath = self.y_8k[index]
            y_16kpath = self.y_16k[index]
            x,_ = torchaudio.load(x_path, normalize = True)
            y_8k,_ = torchaudio.load(y_8kpath, normalize = True)
            y_16k,_ = torchaudio.load(y_16kpath, normalize = True)


            return index, x, y_8k, y_16k

    def __len__(self):
        length = len(self.x)
        length = np.int32(np.floor(length*self.scale))
        
        return length
