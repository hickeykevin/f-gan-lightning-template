#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.transforms import transforms
import numpy as np
import PIL


# %%
z = torch.randn(64, 100, 1, 1)
array = np.random.randint(255, size=(28, 28),dtype=np.uint8)
array = np.expand_dims(array, axis=0)
image = PIL.Image.fromarray(array)
num_channel=3
transform = transforms.Compose(
    [
            transforms.Resize(64), 
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
            ]
        )

# %%
ii = transform(image)
#%%
class Generator(nn.Module):
    def __init__(self, num_channel, latent_dim):
        super(Generator, self).__init__()
        
        self.block1 = 
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d(64, num_channel, 4, 2, 1, bias=False),
            # state size. (num_channel) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, num_channel, latent_dim):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # input is (num_channel) x 64 x 64
            nn.Conv2d(num_channel, 64, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.main(x)
# %%
g = Generator(3, 100)
generated_image = g.forward(z)

# %%
d = Discriminator(3, 100)
d_prediction_generated = d.forward(generated_image)
d_prediction_real = d.forward(ii)
# %%
