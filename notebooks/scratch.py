#%%
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
    print(x.size())
    x = F.leaky_relu(self.conv_layer(x), 0.01)
    print(x.size())
    x = F.leaky_relu(self.conv_bn_layer_1(x), 0.01)
    print(x.size())
    x = F.leaky_relu(self.conv_bn_layer_2(x), 0.01)
    print(x.size())
    x = F.leaky_relu(self.conv_bn_layer_3(x), 0.01)
    print(x.size())
    x = self.flatten(x)
    print(x.size())
    x = self.fc_layer(x)
    print(x.size())

    return x
# %%
v = V()
x = torch.randn(64, 1, 28, 28)
v.forward(x)



# %%
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
# %%
g = Generator(784, 64, 100)

x = torch.randn(64, 1, 28, 28)
z = torch.randn(64, 100)

g.forward(z).size()
# %%
