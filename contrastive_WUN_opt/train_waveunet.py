import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import torchaudio
import json
import argparse
import librosa
from tqdm import tqdm
sys.path.append("./..")
from model.waveunet import waveunet, MSTFTLoss
from dataset import BWEDataset, BWENoiseDataset
from utils_nn import mu_law_decode, mu_law_encode
sys.path.append("./..")

#======Configureation filte setting======#
parser = argparse.ArgumentParser()
parser.add_argument('--conf', action='append')
args = parser.parse_args()

if args.conf is not None:
    for conf_fname in args.conf:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**json.load(f))

    # Reload arguments to override config file values with command line values
    args = parser.parse_args()

#======Environment setting======#
os.environ["CUDA_VISIBLE_DEVICES"] = args.env
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#======Read model======#
Model = waveunet().to(device)

#======Show #parameters======#
pytorch_total_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)
print(pytorch_total_params)

#======Training setting======#
loss = MSTFTLoss()
MAELoss = nn.L1Loss()
optimizer = optim.Adam(Model.parameters(), lr = 0.0002)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001)
lambda_ = args.Lambda
n_epoch = args.num_epoch
n_batch = args.batch_size
scale = args.dataset_scale

#======STFT setting======#
transform= torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, win_length=256, power=None).to(device)
transform_sinc = torchaudio.transforms.Resample(8000, 16000).to(device)
os.chdir('./..')
traindataset = BWEDataset(type='train', language='ENG', noise=None, snr=args.snr, scale=scale)
trainloader = DataLoader(dataset=traindataset, batch_size=n_batch, shuffle=True)
print('trainset size: %d' %traindataset.__len__())
print('trainset dome!')
cleandataset = BWEDataset(type='valid', language='ENG', noise='clean', snr=args.snr, scale=scale)
cleanloader = DataLoader(dataset=cleandataset, batch_size=n_batch, shuffle=True)
print('clean validset size: %d' %cleandataset.__len__())
print('clean validset done!')
noisydataset = BWEDataset(type='valid', language='ENG', noise='noisy', snr=args.snr, scale=scale)
noisyloader = DataLoader(dataset=noisydataset, batch_size=n_batch, shuffle=True)
print('noisy validset size: %d' %noisydataset.__len__())
print('noisy validset done!')
os.chdir('./contrastive_WUN_opt')

def train(net, loss, optimizer, dataloader):
    running_loss = 0.0
    total_loss = 0.0
    total_loss_ = 0.0
    total_loss_512 = 0.0
    total_loss_1024 = 0.0
    total_loss_2048 = 0.0
    
    for i,(index,x,y) in enumerate(dataloader):

        #zero gradients
        optimizer.zero_grad()

        x = x.data.numpy()
        x = librosa.effects.preemphasis(x)
        # x = mu_law_encode(x)
        # y = mu_law_encode(y)
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        
        #x = x.unsqueeze(1)
        #y = y.unsqueeze(1)

        # Prediction = forward pass
        _,y_pred = net(x)
        
        #Compute MSTFT loss
        #l, l_wav, l_512, l_1024, l_2048 = loss(y_pred, y, lambda_, device)
        l, l_wav, l_512, l_1024, l_2048 = loss(y_pred, y, lambda_, device)
        # l = ASRloss(y_pred, y)
        l.backward()

        # update weight
        optimizer.step()

        running_loss += float(l)
        total_loss += float(l)
        total_loss_ += float(l_wav)
        total_loss_512 += float(l_512)
        total_loss_1024 += float(l_1024)
        total_loss_2048 += float(l_2048)
        n = i
        if i % 10 == 9:    # print every 10mini-batches
            print('Epoch %03d / %03d loss %.3f' % (epoch + 1, i+1, running_loss/10))
            running_loss = 0.0

    return total_loss/n, total_loss_/n, total_loss_512/n, total_loss_1024/n, total_loss_2048/n
def valid(net, loss, dataloader, type_):
    running_loss = 0.0
    total_loss = 0.0
    
    for i,(index,x,y) in enumerate(dataloader):


        x = x.data.numpy()
        x = librosa.effects.preemphasis(x)
        # x = mu_law_encode(x)
        # y = mu_law_encode(y)
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)


        # Prediction = forward pass
        _,y_pred = net(x)
        
        #Compute MSTFT loss
        l,_,_,_,_= loss(y_pred, y, lambda_, device)

        running_loss += float(l)
        total_loss += float(l)
        n = i
        if i % 10 == 9:    # print every 10mini-batches
            print('%s %03d / %03d loss %.3f' % (type_, epoch + 1, i+1, running_loss/10))
            running_loss = 0.0

    return total_loss/n

epoch_loss = []
epoch_loss_ = []
epoch_loss_512 = []
epoch_loss_1024 = []
epoch_loss_2048 = []
clean_loss = []
noisy_loss = []

LOSS_valid = float("inf")
for epoch in range(n_epoch):
    my_lr = scheduler.optimizer.param_groups[0]['lr']
    print("Learning rate: %f" %my_lr)

    Model.train()
    total_loss, total_loss_, total_loss_512, total_loss_1024, total_loss_2048 = train(Model, loss, optimizer, trainloader)
    
    epoch_loss.append(total_loss)
    epoch_loss_.append(total_loss_)
    epoch_loss_512.append(total_loss_512)
    epoch_loss_1024.append(total_loss_1024)
    epoch_loss_2048.append(total_loss_2048)

    Model.eval()
    with torch.no_grad():
        total_loss1 = valid(Model, loss, cleanloader, 'Clean')
        clean_loss.append(total_loss1)
        total_loss2 = valid(Model, loss, noisyloader, 'Noisy')
        noisy_loss.append(total_loss2)
        total_loss = total_loss1+total_loss2
        if total_loss < LOSS_valid:
            LOSS_valid = total_loss
            path = args.model_target_path +'/waveunet_opt_prefilter_v2.pt'
            torch.save(Model.state_dict(), path)
        # scheduler.step(total_loss)
    
    plt.figure()
    plt.plot(epoch_loss,color = 'r',label = 'training loss')
    plt.plot(clean_loss, color = 'g', label = 'clean validation loss')
    plt.plot(noisy_loss, color = 'b', label = 'noisy validation loss')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.savefig(args.fig_name +".png", dpi=300, format = 'png')

    
    plt.figure()
    plt.plot(epoch_loss_,color = 'r',label = 'training loss_wav')
    plt.plot(epoch_loss_512,color = 'g',label = 'training loss_512')
    plt.plot(epoch_loss_1024,color = 'b',label = 'training loss_1024')
    plt.plot(epoch_loss_2048,color = 'y',label = 'training loss_2048')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.savefig(args.fig_name +"_MSTFT.png", dpi=300, format = 'png')

        

    print(epoch_loss)
    print(epoch_loss_)
    print(epoch_loss_512)
    print(epoch_loss_1024)
    print(epoch_loss_2048)
    print(clean_loss)
    print(noisy_loss)
