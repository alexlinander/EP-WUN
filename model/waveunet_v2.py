import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import cdpam

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60")

class DownSamplingLayer(nn.Module):
    'ds'
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

class SQC(nn.Module):
    def __init__(self):
        super(SQC, self).__init__()

        self.main = nn.Sequential(
            nn.MaxPool1d(3, stride=2),
            nn.PReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.PReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.PReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.PReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.PReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.PReLU()
        )
        self.linear = nn.Sequential(

            # nn.Linear(512,256),
            
            nn.Linear(256,64),
            nn.Linear(64,2)
        )
        self.softmax = nn.Softmax(dim=2)
        self.lstm = BidirectionalLSTM(144,288,144)
    def forward(self, input):
        input = input.permute(2,0,1)
        lstm = self.lstm(input)
        lstm = lstm.permute(1,0,2)
        cnn = self.main(lstm)
        cnn = cnn.permute(0,2,1)
        output = self.linear(cnn)
        output = self.softmax(output)
        return lstm, output

class BidirectionalLSTM(nn.Module):
    'fd'

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, num_layers=2,bidirectional=True)
        self.embedding = nn.Linear(2*nHidden , nOut)

    def forward(self, input):
        'ds'
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(recurrent)  # [T * b, nOut]
        return output

class DownsamplingBlock(nn.Module):
    def __init__(self, n_layers=6, channels_interval=24):
        super(DownsamplingBlock, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )
            
        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, input): 
        tmp = []
        o = input

        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            # print('Encoder Layer {} size : {}'.format(i+1, o.size()))
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)

        return o, tmp


class UpsamplingBlock(nn.Module):
    def __init__(self, n_layers=6, channels_interval=24):
        super(UpsamplingBlock, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input1, input2, tmp):
        
        o = input1
        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)
            # print('Decoder Layer {} size : {}'.format(i+1, o.size()))

        o = torch.cat([o, input2], dim=1)
        o = self.out(o)

        return o
        

class MSTFTLoss(nn.Module):
    def __init__(self, ):
        super(MSTFTLoss, self).__init__()
        self.transform_512= torchaudio.transforms.Spectrogram(n_fft=512, hop_length=50, win_length=240)
        self.transform_1024 = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=120, win_length=600)
        self.transform_2048 = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=240, win_length=1200)

    def forward(self, y_pred, y, Lambda):
        #loss_fn = cdpam.CDPAM()
        loss = nn.L1Loss()


        stft_512 = self.transform_512(y)
        stft_1024 = self.transform_1024(y)
        stft_2048 = self.transform_2048(y)

        stft_512_pred = self.transform_512(y_pred)
        stft_1024_pred = self.transform_1024(y_pred)
        stft_2048_pred = self.transform_2048(y_pred)
        stft_512_pred[stft_512==0] = 10**(-20)
        stft_1024_pred[stft_1024==0] = 10**(-20)
        stft_2048_pred[stft_2048==0] = 10**(-20)
        stft_512_pred[stft_512_pred==0] = 10**(-20)
        stft_1024_pred[stft_1024_pred==0] = 10**(-20)
        stft_2048_pred[stft_2048_pred==0] = 10**(-20)

        stft_512[stft_512==0] = 10**(-20)
        stft_1024[stft_1024==0] = 10**(-20)
        stft_2048[stft_2048==0] = 10**(-20)

        stft_512 = torch.log(stft_512)
        stft_1024 = torch.log(stft_1024)
        stft_2048 = torch.log(stft_2048)
        stft_512_pred = torch.log(stft_512_pred)
        stft_1024_pred = torch.log(stft_1024_pred)
        stft_2048_pred = torch.log(stft_2048_pred)

        l_512 = loss(stft_512, stft_512_pred)
        l_1024 = loss(stft_1024, stft_1024_pred)
        l_2048 = loss(stft_2048, stft_2048_pred)

        l_wav = Lambda*loss(y_pred, y)
        y = y.squeeze()
        y_pred = y_pred.squeeze()

        l = l_wav+l_512+l_1024+l_2048

        return l, l_wav, l_512, l_1024, l_2048

class negative_set_define(nn.Module):

    def __init__(self, ):
        super(negative_set_define, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, anchor, negative):
        negative_refine = torch.zeros(negative.size())
        for i,a in enumerate(anchor):
            loss = torch.zeros(len(anchor))
            for j,n in enumerate(negative):
                loss[j] = self.loss(a,n)
            index = torch.argmax(loss)
            negative_refine[i] = negative[index]

        return negative_refine