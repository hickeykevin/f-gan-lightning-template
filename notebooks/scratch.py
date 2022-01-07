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
import math
from src.models.modules.walter_work import *
from pytorch_lightning import LightningModule
import torch
from torchmetrics import Accuracy

#%%

        
PARAM = {
    "dataset" : 'mnist',#Dataset. Options: cifar10, mnist 
    "exp_name" : 'mnist_score_test',#Name of experiment.
    'div' : 'JSD',#Divergence to use with f-gan. Choices: 'JSD', 'SQH', 'GAN, 'KLD', 'RKL', 'CHI', 'Wasserstein'
    'model' : 'CNN',#Backbone model. Options: DCGAN, DCGAN_128, CNN
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
    'use_noise' : True,#Whether to convolve noise. Boolean. 
    'noise_bandwidth' : 0.01,#covariance of convolved noise. Only applied if use_noise=True. 
    'noise_annealing' : 1.,#decay factor for convolved noise. Only applied if use_noise=True. 
    #######Wasserstein GAN############
    'c' : 0.01,#Weight clipping for Wasserstein GAN. Only applied if div = Wasserstein.
    #######Discriminator Regularization######
    'use_disc_reg' : False,#Whether to use discriminator regularization
    'reg_gamma' : 0.1,#Weight for regularization term
    'reg_annealing' : 1.,#TODO: Fix this to be like in paper
    ###########
    'nwork' : 1,
}

if PARAM['model'] == 'DCGAN_128':
    PARAM['imsize']  = 128# Spatial size of training images. 
else:
    PARAM['imsize']  = 64

SIZES = set_imsize_divisions(PARAM)



class WalterGAN(LightningModule):
    def __init__(
        self,
        div: str = "JSD",
        nc: int = 1,
        nz: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        use_noise: bool = True,
        noise_bandwith: float = 0.01,
        noise_annealing: float = 1.,
        c: float = 0.01,
        use_disc_reg: bool = False,
        reg_gama: float = 0.1,
        reg_annealing: float = 1.,
        ):
                
        super().__init__()
        self.div = div
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.use_noise = use_noise
        self.noise_bandwith = noise_bandwith
        self.noise_annealing = noise_annealing
        self.c = c
        self.use_disc_reg = use_disc_reg
        self.reg_gama = reg_gama
        self.reg_annealing = reg_annealing

        #TODO ORGANIZE THIS!!!
        self.params = PARAM
        self.layer_sizes = SIZES
        
        self.save_hyperparameters()

        if self.params['model'] == 'DCGAN':
            Q_net = Q_DCGAN(self.params).to(self.device)
            V_net = V_DCGAN(self.params).to(self.device)
            
        elif self.params['model'] == 'CNN':
            Q_net = Q_CNN(self.params, self.layer_sizes).to(self.device)
            V_net = V_CNN(self.params, self.layer_sizes).to(self.device)
            
        elif self.params['model'] == 'DCGAN_128':
            Q_net = Q_DCGAN_128(self.params).to(self.device)
            V_net = V_DCGAN_128(self.params).to(self.device)
            
        else:
            Q_net = Q_CNN(self.params, self.layer_sizes).to(self.device)
            V_net = V_CNN(self.params, self.layer_sizes).to(self.device)

        self.generator = Q_net
        self.discriminator = V_net

        if self.params['div'] == 'Wasserstein':
            Q_criterion = Wasserstein_QLOSS(self.params['div'])
            V_criterion = Wasserstein_VLOSS(self.params['div'])
        else:
            Q_criterion = QLOSS(self.params['div'])
            V_criterion = VLOSS(self.params['div'])
        
        self.Q_criterion = Q_criterion
        self.V_criterion = V_criterion
            
        if self.params['use_disc_reg']:
            reg_criterion = REGLOSS(self.params['div'])
            self.reg_criterion = reg_criterion

        self.d_accuracy_on_generated_instances = Accuracy(num_classes=1)

    def forward(self, z):
        return self.generator.forward(z)

    def on_train_start(self) -> None:
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, _ = batch
    
        z = self.sample_z().type_as(data)
        fake_data = self.generator.forward(z)

        if self.params['use_noise'] == True:
            annealed_bandwidth = self.params['noise_bandwidth']*(self.params['noise_annealing']**self.current_epoch)
            noise_term = torch.randn(data.size()).to(self.device) * annealed_bandwidth
            input_data = data + noise_term
            noise_term = torch.randn(fake_data.size()).to(self.device) * annealed_bandwidth
            input_fake = fake_data + noise_term
        else:
            input_data = data
            input_fake = fake_data

        # Train discriminator
        if optimizer_idx == 1:

            # Discriminator output on real instances
            v = self.discriminator(input_data)
            loss_real = -self.V_criterion(v)
            #loss_real.backward(retain_graph=True)
            
            # Discriminator output on fake instances
            v_fake = self.discriminator(input_fake.detach())
            loss_fake = -self.Q_criterion(v_fake)
            #loss_fake.backward()#maximize F

            loss_V = -(loss_real + loss_fake)

            if self.params['use_disc_reg'] == True:
                loss_V += self.params['reg_gamma']*self.reg_criterion(self.discriminator, input_fake.detach())

            self.log("train/V_loss", loss_V, on_epoch=True)
            return {"loss": loss_V}
        
        # Train generator
        if optimizer_idx == 0:

            # Discrimator output on fake instances
            v_fake = self.discriminator.forward(input_fake)
            loss_Q = -self.V_criterion(v_fake)
            self.log("train/Q_loss", loss_Q, on_epoch=True)
            return {"loss": loss_Q}
            
        
    def on_train_batch_end(self, outputs, batch, batch_idx, unused = 0) -> None:
        if self.params['div'] == 'Wasserstein':
            # Clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.params['c'], self.params['c'])

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.beta1, self.beta2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.beta1, self.beta2))
        return [optimizer_G, optimizer_D]

    def sample_z(self):
        if self.params["model"] == "CNN":
            noise = torch.randn(self.params['bsize'], self.params['nz'])
        else:
            noise = torch.randn(self.params['bsize'], self.params['nz'], 1, 1)

        return noise
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv')!=-1 or classname.find('Linear')!=-1:
            nn.init.normal_(m.weight.data,0.0,0.02)
        elif classname.find('BatchNorm')!=-1:
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)


class QLOSS(nn.Module):
    def __init__(self,divergence):
        super(QLOSS,self).__init__()
        self.conjugate = Conjugate_f(divergence)
        self.activation = Activation_g(divergence)
    def forward(self,v):
        return torch.mean(-self.conjugate(self.activation(v)))

class VLOSS(nn.Module):
    def __init__(self,divergence):
        super(VLOSS,self).__init__()
        self.activation = Activation_g(divergence)
    def forward(self,v):
        return torch.mean(self.activation(v))

data = MNISTDataModule()
data.prepare_data()
data.setup(stage=None)
X, y = next(iter(data.train_dataloader()))

model = WalterGAN()    
Q_criterion = QLOSS(divergence="JSD")
V_criterion = VLOSS(divergence="JSD")

noise = torch.randn(PARAM['bsize'], PARAM['nz'])



# %%
fake_data = model.generator.forward(noise)

v_fake = model.discriminator.forward(fake_data)
loss_Q = -V_criterion(v_fake)

output_real = model.discriminator(X)
loss_real = -V_criterion(output_real)

output_fake = model.discriminator.forward(fake_data)
loss_fake = -Q_criterion(output_fake)

loss_V = -(loss_real + loss_fake)

#%%
trainer.fit(model, datamodule=data)


# %%

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
