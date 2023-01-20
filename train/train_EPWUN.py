import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
sys.path.append("./..")
from model.waveunet_v2 import DownsamplingBlock, UpsamplingBlock, SQC, MSTFTLoss
from dataset import BWENoiseDataset

#======Configureation filte setting======#
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='0', help='GPU number')
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--encoder_path', type=str, default='trained_model/BWE_only/WUN_Encoder.pt', help='encoder path')
parser.add_argument('--decoder_path', type=str, default='trained_model/BWE_only/WUN_Decoder.pt', help='decoder path')
parser.add_argument('--SQC_path', type=str, default='trained_model/EP_WUN/SQC/SQC.pt', help='SQC path')
parser.add_argument('--model_path', type=str, default='trained_model/EP_WUN', help='model save path')
parser.add_argument('--Lambda_wav', type=float, default=800, help='weight of wav loss')
parser.add_argument('--Lambda_trp', type=float, default=8, help='weight of triplet loss')
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
os.chdir('./..')
Encoder = DownsamplingBlock()
Encoder.load_state_dict(torch.load(args.encoder_path))
Encoder = Encoder.to(device)
Encoder2 = DownsamplingBlock()
Encoder2.load_state_dict(torch.load(args.encoder_path))
Encoder2 = Encoder2.to(device)
Decoder = UpsamplingBlock()
Decoder.load_state_dict(torch.load(args.decoder_path))
Decoder = Decoder.to(device)
Model_n = SQC()
Model_n.load_state_dict(torch.load(args.SQC_path))
Model_n = Model_n.to(device)

#======Training setting======#
loss = MSTFTLoss().to(device)
TRPloss = nn.TripletMarginLoss()
optimizer = optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr = 0.0002)
lambda_ = args.Lambda_wav
lambda_TRP = args.Lambda_trp
n_epoch = args.epoch
n_batch = args.batch

#======Load dataset======#
traindataset = BWENoiseDataset(type='train', noise=None)
trainloader = DataLoader(dataset=traindataset, batch_size=n_batch, shuffle=True)
print('trainset size: %d' %traindataset.__len__())
print('trainset done!')
posdataset = BWENoiseDataset(type='train', noise='pos')
posloader = DataLoader(dataset=posdataset, batch_size=n_batch, shuffle=True)
print('positive set size: %d' %posdataset.__len__())
print('positive set done!')
negdataset = BWENoiseDataset(type='train', noise='neg')
negloader = DataLoader(dataset=negdataset, batch_size=n_batch, shuffle=True)
print('negative set size: %d' %negdataset.__len__())
print('negative set done!')
cleandataset = BWENoiseDataset(type='valid', noise='clean')
cleanloader = DataLoader(dataset=cleandataset, batch_size=n_batch, shuffle=True)
print('clean validset size: %d' %cleandataset.__len__())
print('clean validset done!')
noisydataset = BWENoiseDataset(type='valid', noise='noisy')
noisyloader = DataLoader(dataset=noisydataset, batch_size=n_batch, shuffle=True)
print('noisy validset size: %d' %noisydataset.__len__())
print('noisy validset done!')

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

if not os.path.exists('fig'):
    os.makedirs('fig')


def train(net_en1, net_en2, net_de, net_n, loss, optimizer, dataloader, dataloader_p, dataloader_n):
    running_loss = 0.0
    wav_loss = 0.0
    MSTFT_loss = 0.0
    TRP_loss = 0.0
    total_loss = 0.0
    loss_wav = 0.0
    loss_TRP = 0.0
    loss_MSTFT = 0.0

    for i,((_,x,y), (_,x_p,_), (_,x_n,_)) in enumerate(zip(dataloader, dataloader_p, dataloader_n)):
        
        optimizer.zero_grad()

        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)
        x_p = torch.Tensor(x_p).to(device)
        x_n = torch.Tensor(x_n).to(device)

        hidden_p,_ = net_en2(x_p)
        hidden_n,_ = net_en2(x_n)
        h_p,_ = net_n(hidden_p)
        h_n,_ = net_n(hidden_n)

        hidden,tmp = net_en1(x)
        h,_ = net_n(hidden)

        hh = h.permute(0,2,1)
        hidden = hh+hidden

        y_pred = net_de(hidden, x, tmp)
        
        _, l_wav, l_512, l_1024, l_2048 = loss(y_pred, y, lambda_)
        l_MSTFT = l_512+l_1024+l_2048
        l_TRP = lambda_TRP*TRPloss(h,h_p,h_n)
        l = l_wav+l_MSTFT+l_TRP

        l.backward()

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
    running_loss = 0.0
    total_loss = 0.0
    
    for i,(_,x,y) in enumerate(dataloader):

        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)


        hidden,tmp = net_en1(x)
        h,_ = net_n(hidden)

        hh = h.permute(0,2,1)
        hidden = hh+hidden

        y_pred = net_de(hidden, x, tmp)
        
        _, l_wav, l_512, l_1024, l_2048 = loss(y_pred, y, lambda_)
        l_MSTFT = l_512+l_1024+l_2048
        l = l_wav+l_MSTFT


        running_loss += float(l)
        total_loss += float(l)
        n = i
        if i % 10 == 9:    # print every 10mini-batches
            print('%s %03d / %03d loss %.3f' % (type_, epoch + 1, i+1,running_loss/10))
            running_loss = 0.0

    return total_loss/n

epoch_loss = []
epoch_loss_wav = []
epoch_loss_MSTFT = []
epoch_loss_TRP = []

clean_loss = []
noisy_loss = []

LOSS_valid = float("inf")
for epoch in range(n_epoch):

    Encoder.train()
    Encoder2.train()
    Decoder.train()
    Model_n.train()
    total_loss, loss_wav, loss_TRP, loss_MSTFT = train(Encoder, Encoder2, Decoder, Model_n, loss, optimizer, trainloader, posloader, negloader)
    epoch_loss.append(total_loss)
    epoch_loss_wav.append(loss_wav)
    epoch_loss_TRP.append(loss_TRP)
    epoch_loss_MSTFT.append(loss_MSTFT)

    Encoder.eval()
    Encoder2.eval()
    Decoder.eval()
    Model_n.eval()
    with torch.no_grad():

        total_loss1 = valid(Encoder, Encoder2, Decoder, Model_n, loss, cleanloader, 'Clean')
        clean_loss.append(total_loss1)

        total_loss2 = valid(Encoder, Encoder2, Decoder, Model_n, loss, noisyloader, 'Noisy')
        noisy_loss.append(total_loss2)

        if total_loss < LOSS_valid:
            print("======Writing Model!======")
            LOSS_valid = total_loss
            path_encoder = args.model_path +'/EPWUN_Encoder.pt'
            path_decoder = args.model_path +'/EPWUN_Decoder.pt'
            torch.save(Encoder.state_dict(), path_encoder)
            torch.save(Decoder.state_dict(), path_decoder)
    
    plt.figure()
    plt.plot(epoch_loss_wav,color = 'r',label = 'training loss_wav')
    plt.plot(epoch_loss_MSTFT,color = 'g',label = 'training loss_MSTFT')
    plt.plot(epoch_loss_TRP,color = 'b',label = 'training loss_TRP')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("fig/EP_WUN_train.png", dpi=300, format = 'png')

    
    plt.figure()
    plt.plot(clean_loss,color = 'r',label = 'clean loss')
    plt.plot(noisy_loss,color = 'g',label = 'noisy loss')
    plt.legend(loc='upper right')
    plt.title('Loss vs Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("fig/EP_WUN_valid.png", dpi=300, format = 'png')
    

