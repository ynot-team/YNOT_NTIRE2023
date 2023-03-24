import torch
import torch.nn as nn
from utils.blocks_for_ffc_all_orig import MeanShift, IMDN_Module, FourierUnit, FFC, SpectralTransform, WavePool_LL, WavePool_LH, WavePool_HL, WavePool_HH, get_wav, WavePool, WaveUnpool
# from .fft_conv import fft_conv, FFTConv2d

import numpy as np

# def get_wav(in_channels, out_channels, pool=True, group=1):
#     """wavelet decomposition using conv2d"""
#     harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
#     harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
#     harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

#     harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
#     harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
#     harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
#     harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

#     filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
#     filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
#     filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
#     filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

#     if pool:
#         net = nn.Conv2d
#     else:
#         net = nn.ConvTranspose2d

#     LL = net(in_channels, out_channels,
#              kernel_size=2, stride=2, padding=0, bias=False,
#              groups=group)
#     LH = net(in_channels, out_channels,
#              kernel_size=2, stride=2, padding=0, bias=False,
#              groups=group)
#     HL = net(in_channels, out_channels,
#              kernel_size=2, stride=2, padding=0, bias=False,
#              groups=group)
#     HH = net(in_channels, out_channels,
#              kernel_size=2, stride=2, padding=0, bias=False,
#              groups=group)

#     LL.weight.requires_grad = False
#     LH.weight.requires_grad = False
#     HL.weight.requires_grad = False
#     HH.weight.requires_grad = False

#     # LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
#     # LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
#     # HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
#     # HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

#     LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, out_channels, -1, -1)
#     LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, out_channels, -1, -1)
#     HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, out_channels, -1, -1)
#     HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, out_channels, -1, -1)

#     return LL, LH, HL, HH

class FFC14_ST_all_wavelet_front_module4(nn.Module):
    def __init__(self, upscale_factor=4, in_channels=3, num_fea=64, out_channels=3, imdn_blocks=4):
        super(FFC14_ST_all_wavelet_front_module4, self).__init__()
        
        #self.sub_mean = MeanShift()
        #self.add_mean = MeanShift(sign=1)

        # extract features
        # self.fea_conv = nn.Conv2d(in_channels, num_fea, 3, 1, 1)
        # self.fea_conv = FFC(in_channels, num_fea, 3, 0, 1)  # bad
        # self.fea_conv = FFTConv2d(in_channels, num_fea, 128, bias=True)
        # self.fea_conv = FourierUnit(in_channels, num_fea)
        
        self.fea_conv = SpectralTransform(in_channels, num_fea)


        # self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, num_fea)
        # print('self.LL.weight.shape', self.LL.weight.shape)
        # self.fea_conv = self.LL(in_channels, num_fea, 3, 1, 1, False, 1)
        
        # self.pool1 = WavePool(num_fea)
        # self.LL, self.LH, self.HL, self.HH = WavePool(num_fea)

        self.LL = WavePool_LL(num_fea)
        self.LH = WavePool_LH(num_fea)
        self.HL = WavePool_HL(num_fea)
        self.HH = WavePool_HH(num_fea)

        self.waveunpool = WaveUnpool(num_fea, option_unpool='sum')
        # self.LL, self.LH, self.HL, self.HH = get_wav(num_fea)
        # self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

        # if self.option_unpool == 'sum':
        #     return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        # elif self.option_unpool == 'cat5' and original is not None:
        #     return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
    
        # map
        self.IMDN1 = IMDN_Module(num_fea)
        self.IMDN2 = IMDN_Module(num_fea)
        self.IMDN3 = IMDN_Module(num_fea)
        self.IMDN4 = IMDN_Module(num_fea)
        # self.IMDN5 = IMDN_Module(num_fea)
        # self.IMDN6 = IMDN_Module(num_fea)

        self.fuse = nn.Sequential(
            nn.Conv2d(num_fea * imdn_blocks, num_fea, 1, 1, 0),
            nn.LeakyReLU(0.05)
            # nn.GELU()
        )
        # self.LR_conv = nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        self.LR_conv = SpectralTransform(num_fea, num_fea)

        # reconstruct
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_fea, out_channels * (upscale_factor ** 2), 3, 1, 1),
            # SpectralTransform(num_fea, out_channels * (upscale_factor ** 2)),
            nn.PixelShuffle(upscale_factor)
        )
   
    def forward(self, x):
        #x = self.sub_mean(x)

        # print('x.size()', x.size())
        # extract features
        # print(x.shape)  # torch.Size([1, 3, 360, 640])
        x = self.fea_conv(x)  # torch.Size([1, 64, 360, 640])
        # print(x.shape)

        LL1 = self.LL(x)
        LH1 = self.LH(x)
        HL1 = self.HL(x)
        HH1 = self.HH(x)
        # x = self.LL(x) + self.LH(x) + self.HL(x) + self.HH(x)
        # x = torch.cat([self.LL(x), self.LH(x), self.HL(x), self.HH(x), original], dim=1)

        # body map
        out1 = self.IMDN1(x)
        out2 = self.IMDN2(out1)
        out3 = self.IMDN3(out2)
        out4 = self.IMDN4(out3)
        # out5 = self.IMDN5(out4)
        # out6 = self.IMDN6(out5)

        # print(out1.shape, out2.shape, out3.shape, out4.shape, out5.shape, out6.shape)

        # LL6 = self.LL(out6)
        LL4 = self.LL(out4)


        # print(LL1.shape, LH1.shape, HL1.shape, HH1.shape, LL6.shape)
        # x = self.waveunpool(LL6, LH1, HL1, HH1)
        x = self.waveunpool(LL4, LH1, HL1, HH1)  # torch.Size([1, 64, 180, 320]) torch.Size([1, 64, 180, 320]) torch.Size([1, 64, 180, 320]) torch.Size([1, 64, 180, 320]) torch.Size([1, 64, 180, 320])
        # print(x.shape)
        # out = self.LR_conv(self.fuse(torch.cat([out1, out2, out3, out4, out5, out6], dim=1))) + x
        
        
        # out = self.LR_conv(self.fuse(torch.cat([out1, out2, out3, out4, out5, out6], dim=1))) + x  # torch.Size([1, 320, 360, 640]) 
        out = self.LR_conv(self.fuse(torch.cat([out1, out2, out3, out4], dim=1))) + x  # torch.Size([1, 320, 360, 640]) 

        
        # print(self.LR_conv(self.fuse(torch.cat([out1, out2, out3, out4, out5, out6], dim=1))).shape)
        # print(out.shape)    # torch.Size([1, 64, 360, 640])

        # reconstruct
        out = self.upsampler(out)
        #out = self.add_mean(out)

        return out

def srmodel(scale):
    model = FFC14_ST_all_wavelet_front_module4(upscale_factor=scale, in_channels=3, num_fea=64, out_channels=3, imdn_blocks=4)
    
    return model
