import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import torchaudio
import json
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import librosa
sys.path.append("./..")
from model.waveunet_v2 import DownsamplingBlock, UpsamplingBlock, NoiseEncoder, MSTFTLoss
from dataset import BWENoiseDataset
from utils_nn import mu_law_decode, mu_law_encode
from metric import myPESQ
sys.path.append("./..")

#======Configureation filte setting======#
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='0', help='GPU number')
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--encoder_path', type=str, default='trained_model/BWE_only/WUN_Encoder.pt', help='model save path')
parser.add_argument('--model_path', type=str, default='trained_model/EP_WUN/SQC', help='model save path')
args = parser.parse_args()

# if args.conf is not None:
#     for conf_fname in args.conf:
#         with open(conf_fname, 'r') as f:
#             parser.set_defaults(**json.load(f))

#     # Reload arguments to override config file values with command line values
#     args = parser.parse_args()

#======Environment setting======#
os.environ["CUDA_VISIBLE_DEVICES"] = args.env
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#======Read model======#
Encoder = DownsamplingBlock()
Encoder.load_state_dict(torch.load(args.Encoder_path))
Encoder = Encoder.to(device)
Encoder2 = DownsamplingBlock()
Encoder2.load_state_dict(torch.load(args.Encoder_path))
Encoder2 = Encoder2.to(device)
Decoder = UpsamplingBlock()
Decoder.load_state_dict(torch.load(args.Decoder_path))
Decoder = Decoder.to(device)
Model_n = NoiseEncoder()
Model_n.load_state_dict(torch.load(args.noiseencoder_path))
Model_n = Model_n.to(device)

#======Training setting======#
loss = MSTFTLoss()
MAELoss = nn.L1Loss()
MSELoss = nn.MSELoss()
TRPloss = nn.TripletMarginLoss()
Labelloss = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr = 0.0002)
lambda_ = args.Lambda_wav
lambda_TRP = args.Lambda_triplet
n_epoch = args.num_epoch
n_batch = args.batch_size
scale = args.dataset_scale

#======STFT setting======#
transform= torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, win_length=256, power=None).to(device)
transform_sinc = torchaudio.transforms.Resample(8000, 16000).to(device)
os.chdir('./..')

#======Load dataset======#
traindataset = BWENoiseDataset(type='train', language='ENG', noise=None, snr=args.snr, scale=scale)
trainloader = DataLoader(dataset=traindataset, batch_size=n_batch, shuffle=True)
print('trainset size: %d' %traindataset.__len__())
# positivedataset = BWENoiseDataset(type='train', language='ENG', noise='clean', snr=args.snr, scale=scale)
# positiveloader = DataLoader(dataset=positivedataset, batch_size=n_batch, shuffle=True)
# print('positive set size: %d' %positivedataset.__len__())
# negativedataset = BWENoiseDataset(type='train', language='ENG', noise='noisy', snr=args.snr, scale=scale)
# negativeloader = DataLoader(dataset=negativedataset, batch_size=n_batch, shuffle=True)
# print('negative set size: %d' %negativedataset.__len__())
# print('trainset done!')
cleandataset = BWENoiseDataset(type='valid', language='ENG', noise='clean', snr=args.snr, scale=scale)
cleanloader = DataLoader(dataset=cleandataset, batch_size=n_batch, shuffle=True)
print('clean validset size: %d' %cleandataset.__len__())
print('clean validset done!')
noisydataset = BWENoiseDataset(type='valid', language='ENG',noise='noisy', snr=args.snr, scale=scale)
noisyloader = DataLoader(dataset=noisydataset, batch_size=n_batch, shuffle=True)
print('noisy validset size: %d' %noisydataset.__len__())
print('noisy validset done!')
os.chdir('./contrastive_WUN_v2')


#======Load clean/noisy wavefile======#
clean_path = glob('/mnt/Internal/esun_bwe/Valentini_dataset/clean_trainset/*_x/*.wav')
noisy_path = glob('/mnt/Internal/esun_bwe/Valentini_dataset/noisy_trainset/*_x/*.wav')

clean_wav = torch.zeros([len(clean_path),1,16384])
noisy_wav = torch.zeros([len(noisy_path),1,16384])

for i, (clean_p, noisy_p) in enumerate(tqdm(zip(clean_path,noisy_path))):
    w_c,_ = torchaudio.load(clean_p, normalize = True)
    w_n,_ = torchaudio.load(noisy_p, normalize = True)
    clean_wav[i] = w_c
    noisy_wav[i] = w_n

def get_positive_negative(index):
    length = len(clean_wav)
    positive = torch.zeros([len(index), 1, 16384])
    negative = torch.zeros([len(index), 1, 16384])

    index_p = torch.randint(0,length,(len(index),))
    index_n = torch.randint(0,length,(len(index),))
    for i, (x,p,n) in enumerate(zip(index, index_p, index_n)):
            positive[i] = clean_wav[p]
            negative[i] = noisy_wav[n]

    return positive, negative

def train(net_en1, net_en2, net_de, net_n, loss, optimizer, dataloader):
    running_loss = 0.0
    wav_loss = 0.0
    MSTFT_loss = 0.0
    TRP_loss = 0.0
    total_loss = 0.0
    loss_wav = 0.0
    loss_TRP = 0.0
    loss_MSTFT = 0.0

    for i,(index,x,y,label) in enumerate(dataloader):
        #zero gradients
        optimizer.zero_grad()

        x_p, x_n = get_positive_negative(label)
        # label_ = torch.zeros([len(index),2]).to(device)
        # label_[:,1] = 1
        
        x = x.data.numpy()
        x = librosa.effects.preemphasis(x)
        x = mu_law_encode(x)

        x_p = x_p.data.numpy()
        x_p = librosa.effects.preemphasis(x_p)
        x_p = mu_law_encode(x_p)

        x_n = x_n.data.numpy()
        x_n = librosa.effects.preemphasis(x_n)
        x_n = mu_law_encode(x_n)
        y = mu_law_encode(y)


        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)
        x_p = torch.Tensor(x_p).to(device)
        x_n = torch.Tensor(x_n).to(device)

        # Prediction = forward pass
        hidden_p,_ = net_en2(x_p)
        hidden_n,_ = net_en2(x_n)
        _,h_p,_ = net_n(hidden_p)
        _,h_n,_ = net_n(hidden_n)

        hidden,tmp = net_en1(x)
        _,h,_ = net_n(hidden)

        hh = h.permute(0,2,1)



        hidden = hh+ hidden

        y_pred = net_de(hidden, x, tmp)
        
        #Compute MAE loss/ MSTFT loss/ Triplet loss
        _, l_wav, l_512, l_1024, l_2048 = loss(y_pred, y, lambda_, device)
        l_MSTFT = l_512+l_1024+l_2048
        l_TRP = lambda_TRP*TRPloss(h,h_p,h_n)
        # l_TRP = lambda_TRP*Labelloss(label_pred, label_)
        l = l_wav+l_MSTFT+l_TRP

        l.backward()

        # update weight
        optimizer.step()

        

        running_loss += float(l)
        wav_loss += float(l_wav)
        TRP_loss += float(l_TRP)
        MSTFT_loss += float(l_MSTFT)
        total_loss += float(l)
        loss_wav += float(l_wav)
        loss_TRP += float(l_TRP)
        loss_MSTFT += float(l_MSTFT)
        n = i
        if i % 10 == 9:    # print every 10mini-batches
            print('Epoch %03d / %03d loss %.3f' % (epoch + 1, i+1, running_loss/10))
            print('Epoch %03d / %03d wav loss %.3f' % (epoch + 1, i+1, wav_loss/10))
            print('Epoch %03d / %03d MSTFT loss %.3f' % (epoch + 1, i+1, MSTFT_loss/10))
            print('Epoch %03d / %03d TRP loss %.3f' % (epoch + 1, i+1, TRP_loss/10))
            running_loss = 0.0
            wav_loss = 0.0
            MSTFT_loss = 0.0
            TRP_loss = 0.0

    return total_loss/n, loss_wav/n, loss_TRP/n, loss_MSTFT/n

def valid(net_en1, net_en2, net_de, net_n, loss, dataloader, type_):
    pesq = 0.0
    running_pesq = 0.0
    
    for i,(index,x,y,label) in enumerate(dataloader):
        pesq_ = []
        
        x = x.data.numpy()
        x = librosa.effects.preemphasis(x)
        x = mu_law_encode(x)

        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        # Prediction = forward pass

        hidden,tmp = net_en1(x)
        _,hh,_ = net_n(hidden)

        hh = hh.permute(0,2,1)
        hidden = hh+hidden

        y_pred = net_de(hidden, x, tmp)
        y_pred = y_pred.data.cpu().numpy()
        y = y.data.cpu().numpy()
        y_pred = mu_law_decode(y_pred)

        for w,r in zip(y_pred, y):
            p = myPESQ(w[0], r[0])
            if p != -7:
                pesq_.append(p)

        pesq += float(np.mean(pesq_))
        running_pesq += float(np.mean(pesq_))
        n = i
        if i % 10 == 9:    # print every 10mini-batches
            print('%s %03d / %03d PESQ %.3f' % (type_, epoch + 1, i+1,running_pesq/10))
            running_pesq = 0.0

    return pesq/n

epoch_loss = []
epoch_loss_wav = []
epoch_loss_MSTFT = []
epoch_loss_TRP = []

clean_pesq = []
noisy_pesq = []
average_loss = []
PESQ_valid = 0.0
for epoch in range(n_epoch):

    Encoder.train()
    Encoder2.train()
    Decoder.train()
    Model_n.train()
    total_loss, loss_wav, loss_TRP, loss_MSTFT = train(Encoder, Encoder2, Decoder, Model_n, loss, optimizer, trainloader)
    epoch_loss.append(total_loss)
    epoch_loss_wav.append(loss_wav)
    epoch_loss_TRP.append(loss_TRP)
    epoch_loss_MSTFT.append(loss_MSTFT)

    Encoder.eval()
    Encoder2.eval()
    Decoder.eval()
    Model_n.eval()
    with torch.no_grad():

        pesq1 = valid(Encoder, Encoder2, Decoder, Model_n, loss, cleanloader, 'Clean')
        clean_pesq.append(pesq1)

        pesq2 = valid(Encoder, Encoder2, Decoder, Model_n, loss, noisyloader, 'Noisy')
        noisy_pesq.append(pesq2)

        average_loss.append(pesq2)
        if pesq2> PESQ_valid and args.write_model:
            print("======Writing Model!======")
            PESQ_valid = pesq2
            path_encoder = args.model_target_path +'/adversarial_WUN_Encoder_opt_final.pt'
            path_decoder = args.model_target_path +'/adversarial_WUN_Decoder_opt_final.pt'
            torch.save(Encoder.state_dict(), path_encoder)
            torch.save(Decoder.state_dict(), path_decoder)
    
    plt.figure()
    plt.plot(epoch_loss,color = 'r',label = 'training loss')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.fig_name +".png", dpi=300, format = 'png')

    
    plt.figure()
    plt.plot(epoch_loss_wav,color = 'r',label = 'training loss_wav')
    plt.plot(epoch_loss_MSTFT,color = 'g',label = 'training loss_MSTFT')
    plt.plot(epoch_loss_TRP,color = 'b',label = 'training loss_TRP')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.fig_name +"_train.png", dpi=300, format = 'png')

    
    plt.figure()
    plt.plot(clean_pesq,color = 'r',label = 'clean PESQ')
    plt.plot(noisy_pesq,color = 'g',label = 'noisy PESQ')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.fig_name +"_PESQ.png", dpi=300, format = 'png')
    
        

    print(epoch_loss)
    print(clean_pesq)
    print(noisy_pesq)
