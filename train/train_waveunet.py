import sys
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
sys.path.append("./..")
from model.waveunet_v2 import DownsamplingBlock, UpsamplingBlock,  MSTFTLoss
from dataset import BWEDataset

#======Configureation filte setting======#
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='0', help='GPU number')
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--batch', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--model_path', type=str, default='trained_model/BWE_only', help='model save path')
parser.add_argument('--Lambda', type=float, default=200, help='weighted parameter between wav loss and MSTFT loss')
args = parser.parse_args()


#======Environment setting======#
os.environ["CUDA_VISIBLE_DEVICES"] = args.env
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#======Read model======#
Encoder = DownsamplingBlock().to(device)
Decoder = UpsamplingBlock().to(device)

#======Show #parameters======#
encoder_total_params = sum(p.numel() for p in Encoder.parameters() if p.requires_grad)
decoder_total_params = sum(p.numel() for p in Decoder.parameters() if p.requires_grad)
print('Encoder #params: %d' %encoder_total_params)
print('Decoder #params: %d' %decoder_total_params)

#======Training setting======#
loss = MSTFTLoss().to(device)
optimizer = optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr = args.lr)
n_epoch = args.epoch
n_batch = args.batch

os.chdir('./..')
traindataset = BWEDataset(type='train', noise=None)
trainloader = DataLoader(dataset=traindataset, batch_size=n_batch, shuffle=True)
print('trainset size: %d' %traindataset.__len__())
print('trainset dome!')
cleandataset = BWEDataset(type='valid', noise='clean')
cleanloader = DataLoader(dataset=cleandataset, batch_size=n_batch, shuffle=True)
print('clean validset size: %d' %cleandataset.__len__())
print('clean validset done!')
noisydataset = BWEDataset(type='valid', noise='noisy')
noisyloader = DataLoader(dataset=noisydataset, batch_size=n_batch, shuffle=True)
print('noisy validset size: %d' %noisydataset.__len__())
print('noisy validset done!')

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

if not os.path.exists('fig'):
    os.makedirs('fig')


def train(net_en, net_de, loss, optimizer, dataloader):
    running_loss = 0.0
    total_loss = 0.0
    total_loss_ = 0.0
    total_loss_512 = 0.0
    total_loss_1024 = 0.0
    total_loss_2048 = 0.0
    
    for i,(index,x,y) in enumerate(dataloader):

        optimizer.zero_grad()

        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        hidden,tmp = net_en(x)
        y_pred = net_de(hidden, x, tmp)
        
        l, l_wav, l_512, l_1024, l_2048 = loss(y_pred, y, args.Lambda)
        l.backward()

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
def valid(net_en, net_de, loss, dataloader, type_):
    running_loss = 0.0
    total_loss = 0.0
    
    for i,(index,x,y) in enumerate(dataloader):

        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        hidden,tmp = net_en(x)
        y_pred = net_de(hidden, x, tmp)
        
        l,_,_,_,_= loss(y_pred, y, args.Lambda)

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

    Encoder.train()
    Decoder.train()
    total_loss, total_loss_, total_loss_512, total_loss_1024, total_loss_2048 = train(Encoder, Decoder, loss, optimizer, trainloader)
    
    epoch_loss.append(total_loss)
    epoch_loss_.append(total_loss_)
    epoch_loss_512.append(total_loss_512)
    epoch_loss_1024.append(total_loss_1024)
    epoch_loss_2048.append(total_loss_2048)

    Encoder.eval()
    Encoder.eval()
    with torch.no_grad():
        total_loss1 = valid(Encoder, Decoder, loss, cleanloader, 'Clean')
        clean_loss.append(total_loss1)
        total_loss2 = valid(Encoder, Decoder, loss, noisyloader, 'Noisy')
        noisy_loss.append(total_loss2)
        total_loss = total_loss1+total_loss2
        if total_loss < LOSS_valid:
            LOSS_valid = total_loss
            path_encoder = args.model_path +'/WUN_Encoder.pt'
            path_decoder = args.model_path +'/WUN_Decoder.pt'
            torch.save(Encoder.state_dict(), path_encoder)
            torch.save(Decoder.state_dict(), path_decoder)
    
    plt.figure()
    plt.plot(epoch_loss,color = 'r',label = 'training loss')
    plt.plot(clean_loss, color = 'g', label = 'clean validation loss')
    plt.plot(noisy_loss, color = 'b', label = 'noisy validation loss')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("fig/WUN.png", dpi=300, format = 'png')

    
    plt.figure()
    plt.plot(epoch_loss_,color = 'r',label = 'training loss_wav')
    plt.plot(epoch_loss_512,color = 'g',label = 'training loss_512')
    plt.plot(epoch_loss_1024,color = 'b',label = 'training loss_1024')
    plt.plot(epoch_loss_2048,color = 'y',label = 'training loss_2048')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("fig/WUN_MSTFT.png", dpi=300, format = 'png')


