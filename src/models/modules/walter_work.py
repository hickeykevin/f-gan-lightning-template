from typing import Dict
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import Dataset
import math

# Parameters to define the model.
PARAMS = {
    "dataset" : 'mnist',#Dataset. Options: cifar10, mnist 
    "exp_name" : 'mnist_score_test',#Name of experiment.
    'div' : 'JSD',#Divergence to use with f-gan. Choices: 'JSD', 'SQH', 'GAN, 'KLD', 'RKL', 'CHI', 'Wasserstein'
    'model' : 'DCGAN',#Backbone model. Options: DCGAN, DCGAN_128, CNN
    "bsize" : 128,#Batch size during training.
    'nc' : 1,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 20,# Number of training epochs.
    'lr' : 0.0002,#Learning rate for optimizers
    'beta1' : 0.5,#Beta1 hyperparam for Adam optimizer
    'beta2' : 0.999,#Beta2 hyperparam for Adam optimizer
    #######Convolving Noise#############
    'use_noise' : False,#Whether to convolve noise. Boolean. 
    'noise_bandwidth' : 0.01,#covariance of convolved noise. Only applied if use_noise=True. 
    'noise_annealing' : 1.,#decay factor for convolved noise. Only applied if use_noise=True. 
    #######Wasserstein GAN############
    'c' : 0.01,#Weight clipping for Wasserstein GAN. Only applied if div = Wasserstein.
    #######Discriminator Regularization######
    'use_disc_reg' : False,#Whether to use discriminator regularization
    'reg_gamma' : 0.1,#Weight for regularization term
    'reg_annealing' : 1.,#TODO: Fix this to be like in paper
    ###########
    'nwork' : 1
    }

if PARAMS['model'] == 'DCGAN_128':
    PARAMS['imsize']  = 128# Spatial size of training images. 
else:
    PARAMS['imsize']  = 64


def conv_bn_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    )

def tconv_bn_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    )

def tconv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

def conv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

def fc_layer(in_features,out_features):
    return nn.Linear(in_features,out_features)

def fc_bn_layer(in_features,out_features):
    return nn.Sequential(
        nn.Linear(in_features,out_features),
        nn.BatchNorm1d(out_features)
    )


def set_imsize_divisions(params) -> Dict:

    def conv_out_size_same(size, stride):
        return int(math.ceil(float(size) / float(stride)))
    
    sizes = dict()
    sizes['s_h'], sizes['s_w'] = params["imsize"], params["imsize"]
    sizes['s_h2'], sizes['s_w2'] = conv_out_size_same(sizes['s_h'], 2), conv_out_size_same(sizes['s_w'], 2)
    sizes['s_h4'], sizes['s_w4'] = conv_out_size_same(sizes['s_h2'], 2), conv_out_size_same(sizes['s_w2'], 2)
    sizes['s_h8'], sizes['s_w8'] = conv_out_size_same(sizes['s_h4'], 2), conv_out_size_same(sizes['s_w4'], 2)
    sizes['s_h16'], sizes['s_w16'] = conv_out_size_same(sizes['s_h8'], 2), conv_out_size_same(sizes['s_w8'], 2)
    return sizes

SIZES = set_imsize_divisions(PARAMS)

class Q_CNN(nn.Module):
    def __init__(self, params, layer_sizes):
        super(Q_CNN,self).__init__()
        self.params = params
        self.layer_sizes = layer_sizes
        
        self.projection_z = fc_bn_layer(self.params['nz'],self.layer_sizes['s_h16']*self.layer_sizes['s_w16']*self.params['ngf']*8)
        
        self.theta_params = nn.Sequential(
        tconv_bn_layer(self.params['ngf']*8,self.params['ngf']*4,4,stride=2,padding=1),
                        nn.ReLU(),
        tconv_bn_layer(self.params['ngf']*4,self.params['ngf']*2,4,stride=2,padding=1),
                        nn.ReLU(),
        tconv_bn_layer(self.params['ngf']*2,self.params['ngf'],4,stride=2,padding=1),
                        nn.ReLU(),
        tconv_layer(self.params['ngf'],self.params['nc'],4,stride=2,padding=1),
                        nn.Tanh()
        )
    
    def forward(self, x):
        x = self.projection_z(x)
        x = x.view(-1,self.params['ngf']*8,self.layer_sizes['s_h16'],self.layer_sizes['s_w16'])
        x = F.relu(x)
        #x = F.relu(self.projection_z(x).view(-1,self.params['ngf']*8,self.layer_sizes['s_h16'],self.layer_sizes['s_w16']))
        x =  self.theta_params(x)
        return x
    
class Q_DCGAN(nn.Module):  
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(self.params['nz'], self.params['ngf']*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.params['ngf']*8, self.params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.params['ngf']*4, self.params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.params['ngf']*2, self.params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(self.params['ngf'], self.params['nc'],
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
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Input is the latent vector Z.
        self.tconv0 = nn.ConvTranspose2d(self.params['nz'], self.params['ngf']*16,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(self.params['ngf']*16)
        
        # Input Dimension: (ngf*16) x 4 x 4
        self.tconv1 = nn.ConvTranspose2d(self.params['ngf']*16, self.params['ngf']*8,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.params['ngf']*8, self.params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.params['ngf']*4, self.params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.params['ngf']*2, self.params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(self.params['ngf'], self.params['nc'],
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

"""
input : X ( data )
output : R (scalar) logit

parameters : w
"""
class V_CNN(nn.Module):
    def __init__(self, params, layer_sizes):
        super(V_CNN,self).__init__()
        self.params = params
        self.layer_sizes = layer_sizes
        self.w_params = nn.Sequential (
            conv_layer(self.params['nc'],self.params['ndf'],4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            conv_bn_layer(self.params['ndf'],self.params['ndf']*2,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            conv_bn_layer(self.params['ndf']*2,self.params['ndf']*4,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            conv_bn_layer(self.params['ndf']*4,self.params['ndf']*8,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(1),
            fc_layer(self.params['ndf']*8*self.layer_sizes['s_h16']*self.layer_sizes['s_w16'],1)
        )

    def forward(self, x):
        x = self.w_params(x)
        return x

class V_DCGAN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(self.params['nc'], self.params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(self.params['ndf'], self.params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(self.params['ndf']*2, self.params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(self.params['ndf']*4, self.params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(self.params['ndf']*8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = self.conv5(x)

        return x
    
class V_DCGAN_128(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Input Dimension: (nc) x 128 x 128
        self.conv1 = nn.Conv2d(self.params['nc'], self.params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 64 x 64
        self.conv2 = nn.Conv2d(self.params['ndf'], self.params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.params['ndf']*2)

        # Input Dimension: (ndf*2) x 32 x 32
        self.conv3 = nn.Conv2d(self.params['ndf']*2, self.params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.params['ndf']*4)

        # Input Dimension: (ndf*4) x 16 x 16
        self.conv4 = nn.Conv2d(self.params['ndf']*4, self.params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.params['ndf']*8)

        # Input Dimension: (ndf*8) x 8 x 8
        self.conv5 = nn.Conv2d(self.params['ndf']*8, self.params['ndf']*16,
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.params['ndf']*16)
        
        # Input Dimension: (ndf*8) x 4 x 4
        self.conv6 = nn.Conv2d(self.params['ndf']*16, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)

        x = self.conv6(x)

        return x

class Activation_g(nn.Module):
    def __init__(self,divergence):
        super(Activation_g,self).__init__()
        self.divergence =divergence
    def forward(self,v):
        divergence = self.divergence
        if divergence == "KLD":
            return v
        elif divergence == "RKL":
            return -torch.exp(-v)
        elif divergence == "CHI":
            return v
        elif divergence == "SQH":
            return 1-torch.exp(-v)
        elif divergence == "JSD":
            return torch.log(torch.tensor(2.))-torch.log(1.0+torch.exp(-v))
        elif divergence == "GAN":
            return -torch.log(1.0+torch.exp(-v)) # log sigmoid

class Conjugate_f(nn.Module):
    def __init__(self,divergence):
        super(Conjugate_f,self).__init__()
        self.divergence = divergence
    def forward(self,t):
        divergence= self.divergence
        if divergence == "KLD":
            return torch.exp(t-1)
        elif divergence == "RKL":
            return -1 -torch.log(-t)
        elif divergence == "CHI":
            return 0.25*t**2+t
        elif divergence == "SQH":
            return t/(torch.tensor(1.)-t)
        elif divergence == "JSD":
            return -torch.log(2.0-torch.exp(t))
        elif divergence == "GAN":
            return  -torch.log(1.0-torch.exp(t))

class Conjugate_double_prime(nn.Module):
    def __init__(self,divergence):
        super(Conjugate_double_prime,self).__init__()
        self.divergence = divergence
    def forward(self,v):
        divergence= self.divergence
        if divergence == "KLD":
            return torch.exp(v-1)
        elif divergence == "RKL":
            return 1./(v**2)
        elif divergence == "CHI":
            return 0.5*(v**0)
        elif divergence == "SQH":
            return (2*v)/((1-v)**3) + 2/((1-v)**2)
        elif divergence == "JSD":
            return 2*(torch.exp(v))/((2-torch.exp(v))**2)
        elif divergence == "GAN":
            return  (torch.exp(v))/((1-torch.exp(v))**2)
        
class net_with_flat_input(nn.Module): 
    def __init__(self,net, original_shape):
        super(net_with_flat_input,self).__init__()
        self.net = net
        self.original_shape = original_shape
    def forward(self,x):
        x = x.view(self.original_shape)
        out = self.net(x)
        return out.view(out.shape[0], 1)
    
def compute_delta_x(net, x):
    original_shape = x.shape
    delta_x = torch.autograd.functional.jacobian(net, x, create_graph=True)
    delta_x = torch.diagonal(delta_x, dim1=0, dim2=2).permute(2, 0, 1)
    delta_x = delta_x.reshape(delta_x.shape[0], delta_x.shape[2])
    return delta_x

# class combined_T(nn.Module):
#     def __init__(self,disc, activation):
#         super(combined_T,self).__init__()
#         self.disc = disc
#         self.activation = activation
#     def forward(self,x):
#         return self.activation(self.disc(x))

class combined_T(nn.Module):
    def __init__(self,disc, activation, original_shape):
        super(combined_T,self).__init__()
        self.disc = disc
        self.activation = activation
        self.original_shape = original_shape
    def forward(self,x):
        x = x.view(self.original_shape)
        out = self.activation(self.disc(x))
        return out.view(out.shape[0], 1)

class REGLOSS(nn.Module):
    def __init__(self,divergence):
        super(REGLOSS,self).__init__()
        self.activation = Activation_g(divergence)
        self.f_double = Conjugate_double_prime(divergence)
        
    def forward(self, disc, fake_data):
        flat_data = fake_data.view(fake_data.shape[0], -1)
        T = combined_T(disc, self.activation, fake_data.shape)
        grad_T_logits = compute_delta_x(T, flat_data)
        grad_T_logits_norm = torch.norm(grad_T_logits, dim=1, keepdim=True)
        t = T(flat_data)
        f_dprime_t = self.f_double(t)
        disc_regularizer = grad_T_logits_norm*(f_dprime_t**2)
        disc_regularizer = torch.mean(disc_regularizer)
        return disc_regularizer

class VLOSS(nn.Module):
    def __init__(self,divergence):
        super(VLOSS,self).__init__()
        self.activation = Activation_g(divergence)
    def forward(self,v):
        return torch.mean(self.activation(v))
    
class Wasserstein_VLOSS(nn.Module):
    def __init__(self,divergence="Wasserstein"):
        super(Wasserstein_VLOSS,self).__init__()
    def forward(self,v):
        return torch.mean(v)

class QLOSS(nn.Module):
    def __init__(self,divergence):
        super(QLOSS,self).__init__()
        self.conjugate = Conjugate_f(divergence)
        self.activation = Activation_g(divergence)
    def forward(self,v):
        return torch.mean(-self.conjugate(self.activation(v)))
    
class Wasserstein_QLOSS(nn.Module):
    def __init__(self,divergence="Wasserstein"):
        super(Wasserstein_QLOSS,self).__init__()
    def forward(self,v):
        return -torch.mean(v)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1 or classname.find('Linear')!=-1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)