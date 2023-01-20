import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
sys.path.append("./..")
from model.waveunet import DownsamplingBlock, SQC
from dataset import SQCDataset

#======Configureation filte setting======#
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='0', help='GPU number')
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--encoder_path', type=str, default='trained_model/BWE_only/WUN_Encoder.pt', help='encoder path')
parser.add_argument('--model_path', type=str, default='trained_model/EP_WUN/SQC', help='model save path')
args = parser.parse_args()

#======Environment setting======#
os.environ["CUDA_VISIBLE_DEVICES"] = args.env
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#======Read model======#
os.chdir('./..')
Encoder = DownsamplingBlock().to(device)
Encoder.load_state_dict(torch.load(args.encoder_path))
Encoder = Encoder.to(device)
Model2 = SQC().to(device)

#======Training setting======#
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model2.parameters(), lr = args.lr)
n_epoch = args.epoch
n_batch = args.batch

#======Dataset loading======#
traindataset = SQCDataset(type='train', noise=None)
trainloader = DataLoader(dataset=traindataset, batch_size=n_batch, shuffle=True)
print('trainset size: %d' %traindataset.__len__())
print('trainset done!')
cleandataset = SQCDataset(type='valid', noise='clean')
cleanloader = DataLoader(dataset=cleandataset, batch_size=n_batch, shuffle=True)
print('clean validset size: %d' %cleandataset.__len__())
print('clean validset done!')
noisydataset = SQCDataset(type='valid', noise='noisy')
noisyloader = DataLoader(dataset=noisydataset, batch_size=n_batch, shuffle=True)
print('noisy validset size: %d' %noisydataset.__len__())
print('noisy validset done!')

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

if not os.path.exists('fig'):
    os.makedirs('fig')

def train(net1, net2, loss, optimizer, dataloader):
    running_loss = 0.0
    total_loss = 0.0
    
    for i,(index,x,y) in enumerate(dataloader):

        optimizer.zero_grad()

        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        hidden,_ = net1(x)
        _, y_pred = net2(hidden)
        y_pred = y_pred.squeeze(1)
        
        l = loss(y_pred, y)
        l.backward()

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

        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        # Prediction = forward pass
        hidden,_ = net1(x)
        _,y_pred = net2(hidden)
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
        if total_loss < LOSS_valid:
            print("======Writing Model!======")
            LOSS_valid = total_loss
            path = args.model_path +'/SQC.pt'
            torch.save(Model2.state_dict(), path)
    
    plt.figure()
    plt.plot(epoch_loss,color = 'r',label = 'training loss')
    plt.plot(noisy_loss, color = 'g', label = 'noisy validation loss')
    plt.plot(clean_loss, color = 'b', label = 'clean validation loss')
    plt.plot(average_loss, color = 'y', label = 'average loss')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("fig/SQC.png", dpi=300, format = 'png')
        
    plt.figure()
    plt.plot(noisy_acc, color = 'g', label = 'noisy accuracy')
    plt.plot(clean_acc, color = 'b', label = 'clean accuracy')
    plt.plot(average_acc, color = 'y', label = 'average accuracy')
    plt.legend(loc='upper right')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig("fig/SQC_acc.png", dpi=300, format = 'png')
