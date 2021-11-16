
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

