
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.loss_modules import conv_bn_layer, tconv_bn_layer, tconv_layer, conv_layer, fc_layer, fc_bn_layer

class V_CNN(nn.Module):
    def __init__(self, nc, ndf, height, width):
        super(V_CNN,self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ndf = ndf
        self.w_params = nn.Sequential (
            conv_layer(self.nc, self.ndf,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            conv_bn_layer(self.ndf,self.ndf*2,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            conv_bn_layer(self.ndf*2,self.ndf*4,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            conv_bn_layer(self.ndf*4,self.ndf*8,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(1),
            fc_layer(self.ndf * 8 * height * width,1)
        )

    def forward(self, x):
        x = self.w_params(x)
        return x

class V_DCGAN(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.nc = nc
        self.ndf = ndf

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(self.nc, self.ndf,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ndf*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = self.conv5(x)

        return x
    
class V_DCGAN_128(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.nc = nc
        self.ndf = ndf

        # Input Dimension: (nc) x 128 x 128
        self.conv1 = nn.Conv2d(self.nc, self.ndf,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 64 x 64
        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ndf*2)

        # Input Dimension: (ndf*2) x 32 x 32
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf*4)

        # Input Dimension: (ndf*4) x 16 x 16
        self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf*8)

        # Input Dimension: (ndf*8) x 8 x 8
        self.conv5 = nn.Conv2d(self.ndf*8, self.ndf*16,
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.ndf*16)
        
        # Input Dimension: (ndf*8) x 4 x 4
        self.conv6 = nn.Conv2d(self.ndf*16, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)

        x = self.conv6(x)

        return x