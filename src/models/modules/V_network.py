
import torch
import torch.nn as nn
import torch.nn.functional as F

class V(nn.Module):
  def __init__(self):
    super(V, self).__init__()

    self.c_dim=1
    self.df_dim = 64
    
    self.conv_layer = nn.Conv2d(self.c_dim, self.df_dim, 4, stride=2, padding=1, bias=False)
    
    self.conv_bn_layer_1 = nn.Sequential(
        nn.Conv2d(self.df_dim, self.df_dim*2, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.df_dim*2))

    self.conv_bn_layer_2 = nn.Sequential(
        nn.Conv2d(self.df_dim*2, self.df_dim*4, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.df_dim*4)) 

    self.conv_bn_layer_3 = nn.Sequential(
        nn.Conv2d(self.df_dim*4, self.df_dim*8, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(self.df_dim*8))

    self.flatten = nn.Flatten(1)
    self.fc_layer = nn.Linear(self.df_dim*8, 1, bias=False)

  def forward(self, x):
    x = F.leaky_relu(self.conv_layer(x), 0.01)
    x = F.leaky_relu(self.conv_bn_layer_1(x), 0.01)
    x = F.leaky_relu(self.conv_bn_layer_2(x), 0.01)
    x = F.leaky_relu(self.conv_bn_layer_3(x), 0.01)
    x = self.flatten(x)
    x = self.fc_layer(x)
    return x

class Discriminator(nn.Module):
    """ Discriminator. Input is an image (real or generated),
    output is P(generated).
    """
    def __init__(self, image_size, hidden_dim, output_dim):
        super().__init__()
        self.linear_one = nn.Linear(image_size, hidden_dim)
        self.linear_two = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.linear(x))
        x = self.linear_two(x)
        return x


class DiscriminatorMultipleLayers(nn.Module):
    """ Discriminator. Input is an image (real or generated),
    output is P(generated).
    """
    def __init__(self, image_size, hidden_dim, output_dim):
        super().__init__()
        self.linear_one = nn.Linear(image_size, hidden_dim)
        self.linear_two = nn.Linear(hidden_dim, hidden_dim)
        self.linear_three = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.linear_one(x))
        x = F.elu(self.linear_two(x))
        x = self.linear_three(x)
        return x
#taken from https://github.com/shayneobrien/generative-models/blob/74fbe414f81eaed29274e273f1fb6128abdb0ff5/src/f_gan.py