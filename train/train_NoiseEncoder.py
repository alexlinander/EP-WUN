import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from tqdm import tqdm
import librosa
sys.path.append("./..")
from model.waveunet_v2 import DownsamplingBlock, UpsamplingBlock, NoiseEncoder
from dataset import NoiseEncoderDataset
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
Encoder = DownsamplingBlock().to(device)
Encoder.load_state_dict(torch.load(args.Encoder_path))
Encoder = Encoder.to(device)
Model2 = NoiseEncoder().to(device)

#======Training setting======#
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model2.parameters(), lr = args.learning_rate)
n_epoch = args.num_epoch
n_batch = args.batch_size
scale = args.dataset_scale

#======Dataset loading======#
os.chdir('./..')
traindataset = NoiseEncoderDataset(type='train', language='ENG', noise=None, snr=args.snr, scale=scale)
trainloader = DataLoader(dataset=traindataset, batch_size=n_batch, shuffle=True)
print('trainset size: %d' %traindataset.__len__())
print('trainset done!')
cleandataset = NoiseEncoderDataset(type='valid', language='ENG', noise='clean', snr=args.snr, scale=scale)
cleanloader = DataLoader(dataset=cleandataset, batch_size=n_batch, shuffle=True)
print('clean validset size: %d' %cleandataset.__len__())
print('clean validset done!')
noisydataset = NoiseEncoderDataset(type='valid', language='ENG',noise='noisy', snr=args.snr, scale=scale)
noisyloader = DataLoader(dataset=noisydataset, batch_size=n_batch, shuffle=True)
print('noisy validset size: %d' %noisydataset.__len__())
print('noisy validset done!')
os.chdir('./contrastive_WUN_v2')

def train(net1, net2, loss, optimizer, dataloader):
    running_loss = 0.0
    total_loss = 0.0
    
    for i,(index,x,y) in enumerate(dataloader):
        #zero gradients
        optimizer.zero_grad()

        x = x.data.numpy()
        x = librosa.effects.preemphasis(x)
        x = mu_law_encode(x)
        y = mu_law_encode(y)
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        # Prediction = forward pass
        hidden,_ = net1(x)
        _,_, y_pred = net2(hidden)
        y_pred = y_pred.squeeze(1)
        
        l = loss(y_pred, y)
        l.backward()

        # update weight
        optimizer.step()

        running_loss += float(l)
        total_loss += float(l)
        n = i
        if i % 10 == 9:    # print every 10mini-batches
            print('Epoch %03d / %03d loss %.3f' % (epoch + 1, i+1, running_loss/10))
            running_loss = 0.0

    return total_loss/n

def valid(net1, net2, loss, dataloader, type_):
    running_loss = 0.0
    total_loss = 0.0
    accuracy = []
    
    for i,(index,x,y) in enumerate(dataloader):
        acc = torch.zeros([len(index)])

        x = x.data.numpy()
        x = librosa.effects.preemphasis(x)
        x = mu_law_encode(x)
        y = mu_law_encode(y)
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        # Prediction = forward pass
        hidden,_ = net1(x)
        _,_,y_pred = net2(hidden)
        y_pred = y_pred.squeeze(1)
        
        #Compute MSTFT loss
        l= loss(y_pred, y)

        running_loss += float(l)
        total_loss += float(l)

        y_pred[y_pred<0.5] = 0
        y_pred[y_pred>=0.5] = 1
        acc[y_pred[:,1]==y[:,1]] = 1
        acc[y_pred[:,1]!=y[:,1]] = 0
        accuracy.append(torch.mean(acc))
    
        n = i
        if i % 10 == 9:    # print every 10mini-batches
            print('%s %03d / %03d loss %.3f' % (type_, epoch + 1, i+1, running_loss/10))
            running_loss = 0.0

    return total_loss/n, np.mean(accuracy)

epoch_loss = []
clean_loss = []
noisy_loss = []
average_loss = []
clean_acc = []
noisy_acc = []
average_acc = []

LOSS_valid = float("inf")
for epoch in range(n_epoch):

    Encoder.train()
    Model2.train()
    total_loss = train(Encoder, Model2, loss, optimizer, trainloader)
    
    epoch_loss.append(total_loss)

    Model2.eval()
    with torch.no_grad():
        total_loss1, acc1 = valid(Encoder, Model2, loss, cleanloader, 'Clean')
        print('Epoch %d Clean Accuracy: %3f' % (epoch+1, acc1))
        clean_loss.append(total_loss1)
        clean_acc.append(acc1)

        total_loss2, acc2 = valid(Encoder, Model2, loss, noisyloader, 'Noisy')
        print('Epoch %d Noisy Accuracy: %3f' % (epoch+1, acc2))
        noisy_loss.append(total_loss2)
        noisy_acc.append(acc2)

        average_loss.append((total_loss1+total_loss2)/2)
        average_acc.append((acc1+acc2)/2)
        total_loss = total_loss1+total_loss2
        if total_loss < LOSS_valid and args.write_model:
            print("======Writing Model!======")
            LOSS_valid = total_loss
            path = args.model_target_path +'/NoiseEcoder_opt_final.pt'
            torch.save(Model2.state_dict(), path)
    
    plt.figure()
    plt.plot(epoch_loss,color = 'r',label = 'training loss')
    plt.plot(noisy_loss, color = 'g', label = 'noisy validation loss')
    plt.plot(clean_loss, color = 'b', label = 'clean validation loss')
    plt.plot(average_loss, color = 'y', label = 'average loss')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.savefig(args.fig_name +".png", dpi=300, format = 'png')
        
    plt.figure()
    plt.plot(noisy_acc, color = 'g', label = 'noisy accuracy')
    plt.plot(clean_acc, color = 'b', label = 'clean accuracy')
    plt.plot(average_acc, color = 'y', label = 'average accuracy')
    plt.legend(loc='upper right')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('accuracy')
    plt.ylabel('epoch')
    plt.savefig(args.fig_name +"_acc.png", dpi=300, format = 'png')

    print(epoch_loss)
    print(clean_loss)
    print(noisy_loss)
