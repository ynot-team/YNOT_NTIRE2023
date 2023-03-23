import torch
import torch.nn as nn
import sys
import math
import numpy as np

class MeanShift(nn.Conv2d):
    def __init__(self, mean=[0.4488, 0.4371, 0.4040], std=[1.0, 1.0, 1.0], sign=-1):
        super(MeanShift, self).__init__(3, 3, 1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


class CA(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(CA, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, 1, 0, bias=False)
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, 1, 0, bias=False)
        
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_value = self.avg(x)
        max_value = self.max(x)

        avg_out = self.fc2(self.act(self.fc1(avg_value)))
        max_out = self.fc2(self.act(self.fc1(max_value)))
        
        out = avg_out + max_out
        
        return self.sigmoid(out)

class ResBlock(nn.Module):
    def __init__(self, num_fea=64):
        super(ResBlock, self).__init__()
        self.res_conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )

    def forward(self, x):
        out = self.res_conv(x)

        return out + x

class UpSampler(nn.Module):
    def __init__(self, upscale_factor=2, num_fea=64):
        super(UpSampler, self).__init__()
        if (upscale_factor & (upscale_factor-1)) == 0: # upscale_factor = 2^n
            m = []
            for i in range(int(math.log(upscale_factor, 2))):
                m.append(nn.Conv2d(num_fea, num_fea * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
            self.upsample = nn.Sequential(*m)

        elif upscale_factor == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_fea, num_fea * 9, 3, 1, 1),
                nn.PixelShuffle(3)
            )
        else:
            raise NotImplementedError('Error upscale_factor in Upsampler')

    def forward(self, x):
        return self.upsample(x)


def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])

def std(x):
    assert(x.dim() == 4)
    x_mean = mean_channels(x)
    x_var = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.shape[2] * x.shape[3])
    return x_var.pow(0.5)

class CCA(nn.Module):
    def __init__(self, num_fea, reduction=16):
        super(CCA, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.std = std
        
        self.atten_conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea // reduction, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(num_fea // reduction, num_fea, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        atten = self.avg(x) + self.std(x)
        atten = self.atten_conv(atten)

        return x * atten

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        # channels calculation for local and global branches
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.f_l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.f_l_g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.f_g_l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.f_g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.f_l(x_l) + self.f_g_l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.f_l_g(x_l) + self.f_g(x_g)

        return out_xl, out_xg


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()
        
        # (batch, c, h, w/2+1, 2)
        try:
            ffted = torch.rfft(x, signal_ndim=2, normalized=True)
        except:
            ffted = torch.fft.rfft(x, signal_ndim=2, normalized=True)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        try:
            output = torch.irfft(ffted, signal_ndim=2,
                                signal_sizes=r_size[2:], normalized=True)
        except:
            output = torch.fft.irfft(ffted, signal_ndim=2,
                                signal_sizes=r_size[2:], normalized=True)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output    

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)


    # LL.weight.requires_grad = False
    # LH.weight.requires_grad = False
    # HL.weight.requires_grad = False
    # HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError


class WavePool_LL(nn.Module):
    def __init__(self, in_channels,):
        super(WavePool_LL, self).__init__()
        self.LL, _, _, _ = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x)

class WavePool_LH(nn.Module):
    def __init__(self, in_channels):
        super(WavePool_LH, self).__init__()
        _, self.LH, _, _ = get_wav(in_channels)

    def forward(self, x):
        return self.LH(x)

class WavePool_HL(nn.Module):
    def __init__(self, in_channels):
        super(WavePool_HL, self).__init__()
        _, _, self.HL, _ = get_wav(in_channels)

    def forward(self, x):
        return self.HL(x)

class WavePool_HH(nn.Module):
    def __init__(self, in_channels):
        super(WavePool_HH, self).__init__()
        _, _, _, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.HH(x)


class IMDN_Module(nn.Module):
    def __init__(self, num_fea, distill_ratio=0.25):
        super(IMDN_Module, self).__init__()
        self.distilled_channels = int(num_fea * distill_ratio)
        self.remain_channels = int(num_fea - self.distilled_channels)
        self.conv1 = nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.remain_channels, num_fea, 3, 1, 1)
        self.conv3 = nn.Conv2d(self.remain_channels, num_fea, 3, 1, 1)
        self.conv4 = nn.Conv2d(self.remain_channels, self.distilled_channels, 3, 1, 1)
#         self.conv1 = FourierUnit(num_fea, num_fea)
#         self.conv2 = FourierUnit(self.remain_channels, num_fea)
#         self.conv3 = FourierUnit(self.remain_channels, num_fea)
#         self.conv4 = FourierUnit(self.remain_channels, self.distilled_channels)
        self.act = nn.LeakyReLU(0.05)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)
        self.cca = CCA(num_fea)

    def forward(self, x):
        out1 = self.act(self.conv1(x))
        d1, r1 = torch.split(out1, (self.distilled_channels, self.remain_channels), dim=1)
        out2 = self.act(self.conv2(r1))
        d2, r2 = torch.split(out2, (self.distilled_channels, self.remain_channels), dim=1)
        out3 = self.act(self.conv3(r2))
        d3, r3 = torch.split(out3, (self.distilled_channels, self.remain_channels), dim=1)
        d4 = self.act(self.conv4(r3))
        out = torch.cat([d1, d2, d3, d4], dim=1)
        out = self.cca(torch.cat([d1, d2, d3, d4], dim=1))
        out = self.fuse(out)

        return out + x
