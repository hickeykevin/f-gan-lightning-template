import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.loss_modules import conv_bn_layer, tconv_bn_layer, tconv_layer, conv_layer, fc_layer, fc_bn_layer


class Q_CNN(nn.Module):
    def __init__(self, nz, ngf, nc, height=4, width=4, ):
        super(Q_CNN,self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.height = height
        self.width = width
        
        self.projection_z = fc_bn_layer(nz,height*width*ngf*8)
        
        self.theta_params = nn.Sequential(
        tconv_bn_layer(self.ngf*8,self.ngf*4,4,stride=2,padding=1),
                        nn.ReLU(),
        tconv_bn_layer(self.ngf*4,self.ngf*2,4,stride=2,padding=1),
                        nn.ReLU(),
        tconv_bn_layer(self.ngf*2,self.ngf,4,stride=2,padding=1),
                        nn.ReLU(),
        tconv_layer(self.ngf, nc, 4,stride=2,padding=1),
                        nn.Tanh()
        )
    
    def forward(self, x):
        x = self.projection_z(x)
        x = x.view(-1, self.ngf*8,self.height, self.width)
        x = F.relu(x)
        x =  self.theta_params(x)
        return x
    
class Q_DCGAN(nn.Module):  
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(self.nz, self.ngf*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ngf*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.ngf*8, self.ngf*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ngf*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.ngf*4, self.ngf*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ngf*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.ngf*2, self.ngf,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(self.ngf, self.nc,
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = F.tanh(self.tconv5(x))

        return x
    
class Q_DCGAN_128(nn.Module):  
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        # Input is the latent vector Z.
        self.tconv0 = nn.ConvTranspose2d(self.nz, self.ngf*16,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(self.ngf*16)
        
        # Input Dimension: (ngf*16) x 4 x 4
        self.tconv1 = nn.ConvTranspose2d(self.ngf*16, self.ngf*8,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ngf*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.ngf*8, self.ngf*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ngf*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.ngf*4, self.ngf*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ngf*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.ngf*2, self.ngf,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(self.ngf, self.nc,
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn0(self.tconv0(x)))
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = F.tanh(self.tconv5(x))

        return x

