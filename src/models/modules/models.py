import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

batch_size =64
workers = 2
epochs = 20

latent_dim =100

gf_dim = 64
df_dim = 64

in_h = 64
in_w =64
c_dim = 1

TINY = 1e-6

class Q(nn.Module):
  def __init__(self, latent_space: int, img_shape: tuple, n_classes:int):
    super(Q, self).__init__()
    self.gf_dim = 64
    self.df_dim = 64
    self.c_dim = 1

    #layers
    self.conv_bn_layer_1 = nn.Sequential(
        nn.ConvTranspose2d(self.gf_dim*8, self.gf_dim*4, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.gf_dim*4))

    self.conv_bn_layer_2 = nn.Sequential(
        nn.ConvTranspose2d(self.gf_dim*4, self.gf_dim*2, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.gf_dim*2))

    self.conv_bn_layer_3 = nn.Sequential(
        nn.ConvTranspose2d(self.gf_dim*2, self.gf_dim, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.gf_dim))    
    
    self.tconv_layer = nn.ConvTranspose2d(self.gf_dim, self.c_dim, 4, stride=2, padding=1)
    
    self.projection_z = nn.Sequential(
        nn.Linear(latent_space, 4*4*self.gf_dim*8),
        nn.BatchNorm1d(4*4*self.gf_dim*8)
    )

  def forward(self, x):
    x = F.relu(self.projection_z(x).view(-1, self.gf_dim*8, 4, 4))
    x = F.relu(self.conv_bn_layer_1(x))
    x = F.relu(self.conv_bn_layer_2(x))
    x = F.relu(self.conv_bn_layer_3(x))
    x = F.tanh(self.tconv_layer(x))
    return x


class V(nn.Module):
  def __init__(self):
    super(V, self).__init__()

    self.c_dim=1
    self.df_dim = 64
    
    self.conv_layer = nn.Conv2d(self.c_dim, self.df_dim, 4, stride=2, padding=1)
    
    self.conv_bn_layer_1 = nn.Sequential(
        nn.Conv2d(self.df_dim, self.df_dim*2, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.df_dim*2))

    self.conv_bn_layer_2 = nn.Sequential(
        nn.Conv2d(self.df_dim*2, self.df_dim*4, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.df_dim*4)) 

    self.conv_bn_layer_3 = nn.Sequential(
        nn.Conv2d(self.df_dim*4, self.df_dim*8, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.df_dim*8))

    self.flatten = nn.Flatten(1)
    self.fc_layer = nn.Linear(self.df_dim*8, 1)

  def forward(self, x):
    x = F.leaky_relu(self.conv_layer(x), 0.01)
    x = F.leaky_relu(self.conv_bn_layer_1(x), 0.01)
    x = F.leaky_relu(self.conv_bn_layer_2(x), 0.01)
    x = F.leaky_relu(self.conv_bn_layer_3(x), 0.01)
    x = self.flatten(x)
    x = self.fc_layer(x)
    return x

