import torch
import torch.nn as nn
import torch.nn.functional as F

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