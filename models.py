import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_model import vgg19

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class ResBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottle_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.bottle_conv(x)
        x = self.double_conv(x) + x
        return x / math.sqrt(2)


class Down(nn.Module):
    """Downscaling with stride conv then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            # DoubleConv(in_channels, out_channels)
            ResBlock(in_channels, out_channels)
        )
        

    def forward(self, x):

        x = self.main(x)

        return x

class SDFT(nn.Module):

    def __init__(self, color_dim, channels, kernel_size = 3):
        super().__init__()
        
        # generate global conv weights
        fan_in = channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(color_dim, channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, channels, channels, kernel_size, kernel_size)
        )

    def forward(self, fea, color_style):
        # for global adjustation
        B, C, H, W = fea.size()
        # print(fea.shape, color_style.shape)
        style = self.modulation(color_style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )

        fea = fea.view(1, B * C, H, W)
        fea = F.conv2d(fea, weight, padding=self.padding, groups=B)
        fea = fea.view(B, C, H, W)

        return fea


class UpBlock(nn.Module):
    

    def __init__(self, color_dim, in_channels, out_channels, kernel_size = 3, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels // 2 + in_channels // 8, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_s = nn.Conv2d(in_channels//2, out_channels, 1, 1, 0)

        # generate global conv weights
        self.SDFT = SDFT(color_dim, out_channels, kernel_size)


    def forward(self, x1, x2, color_style):
        # print(x1.shape, x2.shape, color_style.shape)
        x1 = self.up(x1)
        x1_s = self.conv_s(x1)

        x = torch.cat([x1, x2[:, ::4, :, :]], dim=1)
        x = self.conv_cat(x)
        x = self.SDFT(x, color_style)

        x = x + x1_s

        return x


class ColorEncoder(nn.Module):
    def __init__(self, color_dim=512):
        super(ColorEncoder, self).__init__()

        # self.vgg = vgg19(pretrained_path=None)
        self.vgg = vgg19()

        self.feature2vector = nn.Sequential(
            nn.Conv2d(color_dim, color_dim, 4, 2, 2), # 8x8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 4, 2, 2), # 4x4
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1)), # 1x1
            nn.Conv2d(color_dim, color_dim//2, 1), # linear-1
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim//2, color_dim//2, 1), # linear-2
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim//2, color_dim, 1), # linear-3
        )

        self.color_dim = color_dim

    def forward(self, x):
        # x #[0, 1] RGB
        vgg_fea = self.vgg(x, layer_name='relu5_2') # [B, 512, 16, 16]

        x_color = self.feature2vector(vgg_fea[-1]) # [B, 512, 1, 1]

        return x_color


class ColorUNet(nn.Module):
    ### this model output is ab
    def __init__(self, n_channels=1, n_classes=3, bilinear=True):
        super(ColorUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = UpBlock(512, 1024, 512 // factor, 3, bilinear)
        self.up2 = UpBlock(512, 512, 256 // factor, 3, bilinear)
        self.up3 = UpBlock(512, 256, 128 // factor, 5, bilinear)
        self.up4 = UpBlock(512, 128, 64, 5, bilinear)
        self.outc = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 2, 3, 1, 1),
                nn.Tanh()
        )

    def forward(self, x):
        # print(torch.max(x[0]), torch.min(x[0])) #[-1, 1] gray image L
        # print(torch.max(x[1]), torch.min(x[1])) # color vector

        x_color = x[1] # [B, 512, 1, 1]

        x1 = self.inc(x[0]) # [B, 64, 256, 256]
        x2 = self.down1(x1) # [B, 128, 128, 128]
        x3 = self.down2(x2) # [B, 256, 64, 64]
        x4 = self.down3(x3) # [B, 512, 32, 32]
        x5 = self.down4(x4) # [B, 512, 16, 16]

        x6 = self.up1(x5, x4, x_color) # [B, 256, 32, 32]
        x7 = self.up2(x6, x3, x_color) # [B, 128, 64, 64]
        x8 = self.up3(x7, x2, x_color) # [B, 64, 128, 128]
        x9 = self.up4(x8, x1, x_color) # [B, 64, 256, 256]
        x_ab = self.outc(x9)

        return x_ab
