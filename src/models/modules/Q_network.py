import torch
import torch.nn as nn
import torch.nn.functional as F

class Q(nn.Module):
  def __init__(self, latent_space: int):
    super(Q, self).__init__()
    self.latent_space = 100
    self.c_dim = 1
    self.gf_dim = 64
    self.df_dim = 64

    #layers
    self.conv_bn_layer_1 = nn.Sequential(
        nn.ConvTranspose2d(self.gf_dim*8, self.gf_dim*4, 4, stride=2, padding=1, bias=False), #set bias to none/false for each layer
        nn.BatchNorm2d(self.gf_dim*4))

    self.conv_bn_layer_2 = nn.Sequential(
        nn.ConvTranspose2d(self.gf_dim*4, self.gf_dim*2, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.gf_dim*2))

    self.conv_bn_layer_3 = nn.Sequential(
        nn.ConvTranspose2d(self.gf_dim*2, self.gf_dim, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.gf_dim))    
    
    self.tconv_layer = nn.ConvTranspose2d(self.gf_dim, self.c_dim, 4, stride=2, padding=1, bias=False)
    
    self.projection_z = nn.Sequential(
        nn.Linear(self.latent_space, 4*4*self.gf_dim*8, bias=False),
        nn.BatchNorm1d(4*4*self.gf_dim*8)
    )

  def forward(self, x):
    x = F.relu(self.projection_z(x).view(-1, self.gf_dim*8, 4, 4))
    x = F.relu(self.conv_bn_layer_1(x))
    x = F.relu(self.conv_bn_layer_2(x))
    x = F.relu(self.conv_bn_layer_3(x))
    x = F.tanh(self.tconv_layer(x))
    return x

class Generator(nn.Module):
    """ Generator. Input is noise, output is a generated image.
    """
    def __init__(self, image_size, hidden_dim, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.generate = nn.Linear(hidden_dim, image_size)

    def forward(self, x):
        activated = F.relu(self.linear(x))
        generation = torch.sigmoid(self.generate(activated))
        return generation



#taken from https://github.com/shayneobrien/generative-models/blob/74fbe414f81eaed29274e273f1fb6128abdb0ff5/src/f_gan.py