from torch.utils.data import Dataset
import torch
import torchaudio
import numpy as np
from glob import glob

class BWENoiseDataset(Dataset):
    def __init__(self, type='train', noise=None):
        self.type = type
        self.noise = noise

        if type == "train":
            self.x_clean = glob('/dataset/clean_trainset_8k/*.wav')
            self.x_noisy = glob('/dataset/noisy_trainset_8k/*.wav')
            self.y = glob('/dataset/clean_trainset_16k/*.wav')
        elif type == "valid":
            self.x_clean = glob('/dataset/clean_validset_8k/*.wav')
            self.x_noisy = glob('/dataset/noisy_validset_8k/*.wav')
            self.y = glob('/dataset/clean_validset_16k/*.wav')
        elif type == "test":
            self.x_clean = glob('/dataset/clean_testset_8k/*.wav')
            self.x_noisy = glob('/dataset/noisy-testset_8k/*.wav')
            self.y = glob('/dataset/clean_testset_16k/*.wav')

    def __getitem__(self,index):
            
        if self.noise == None:
            if index%2 == 0:
                x_path = self.x_clean[index//2]
            else:
                x_path = self.x_noisy[index//2]
            y_path = self.y[index//2]
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)

        elif self.noise == 'clean':
            x_path = self.x_clean[index]
            y_path = self.y[index]
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)
        elif self.noise == 'noisy':
            x_path = self.x_noisy[index]
            y_path = self.y[index]
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)
        elif self.noise == 'pos':
            x_path = self.x_clean[index//2]
            y_path = self.y[index//2]
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)
        elif self.noise == 'neg':
            x_path = self.x_noisy[index//2]
            y_path = self.y[index//2]
            x,_ = torchaudio.load(x_path, normalize = True)
            y,_ = torchaudio.load(y_path, normalize = True)


        return index, x, y

    def __len__(self):
        length = len(self.x_clean)
        
        if self.noise == None or self.noise == 'pos' or self.noise == 'neg':
            return length*2
        else:
            return length

class BWEDataset(Dataset):
    def __init__(self, type='train', noise=None):
        self.type = type
        self.noise = noise
        
        if type == "train":
            self.x_clean = glob('/dataset/clean_trainset_8k/*.wav')
            self.y_clean = glob('/dataset/clean_trainset_16k/*.wav')
            self.x_noisy = glob('/dataset/noisy_trainset_8k/*.wav')
            self.y_noisy = glob('/dataset/noisy_trainset_16k/*.wav')
        elif type == "valid":
            self.x_clean = glob('/dataset/clean_validset_8k/*.wav')
            self.y_clean = glob('/dataset/clean_validset_16k/*.wav')
            self.x_noisy = glob('/dataset/noisy_validset_8k/*.wav')
            self.y_noisy = glob('/dataset/noisy_validset_16k/*.wav')
        elif type == "test":
            self.x_clean = glob('/dataset/clean_testset_8k/*.wav')
            self.y_clean = glob('/dataset/clean_testset_16k/*.wav')
            self.x_noisy = glob('/dataset/noisy_testset_8k/*.wav')
            self.y_noisy = glob('/dataset/noisy_testset_16k/*.wav')

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
            return length*2
        else:
            return length


class SQCDataset(Dataset):
    def __init__(self, type='train', noise=None):
        self.type = type
        self.noise = noise
    
        if type == "train":
            self.x_clean = glob('/dataset/clean_trainset_8k/*.wav')
            self.x_noisy = glob('/dataset/noisy_trainset_8k/*.wav')
        elif type == "valid":
            self.x_clean = glob('/dataset/clean_validset_8k/*.wav')
            self.x_noisy = glob('/dataset/noisy_validset_8k/*.wav')
        elif type == "test":
            self.x_clean = glob('/dataset/clean_testset_8k/*.wav')
            self.x_noisy = glob('/dataset/noisy_testset_8k/*.wav')

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

