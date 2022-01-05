#%%
%cd ..
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
from src.datamodules.multi_channel_mnist_datamodule import MNISTDataModule
import wandb


# %%
m = MNISTDataModule()
m.prepare_data()
m.setup(stage=None)


# %%
X, y = next(iter(m.train_dataloader()))

#%%
class Generator(nn.Module):
    def __init__(self, num_channel, latent_dim):
        super(Generator, self).__init__()
        
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
        # If image is black & white (channel dimension = 1)
        # Repeat the channel dimension to make it equal to 3 
        # x.shape = (batch_size, num_channels, h, w)
        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)
        return self.main(x)
# %%
g = Generator(3, 100)
z = torch.randn(64, 100, 1, 1)
generated_image = g.forward(z)

# %%
d = Discriminator(3, 100)
d_prediction_generated = d.forward(generated_image)
d_prediction_real = d.forward(X)
# %%
image_predictions_zip = list(zip(generated_image.flatten(1, -1), d_prediction_generated.flatten(1, -1)))
# %%
x = list(zip(generated_image, d_prediction_generated.flatten(1, -1)))

# %%
