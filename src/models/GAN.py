from src.models.modules.Q_network import Q, Generator
from src.models.modules.V_network import V, Discriminator

from src.models.modules.loss_modules import DiscriminatorLoss, GeneratorLoss
from pytorch_lightning import LightningModule

import torch
from collections import OrderedDict
from torchvision.utils import make_grid


class LitFGAN(LightningModule):
  def __init__(
      self,
      latent_dim: int = 100,
      img_size = 784,
      num_classes: int = 10,
      hidden_dim = 64,
      output_dim = 1,
      lr: float = 0.0002,
      chosen_divergence: str = "KLD",
              ):
    
    super().__init__()
    self.latent_dim = latent_dim
    self.img_size = img_size,
    self.batch_size = self.trainer.dataloader.batch_size
    self.num_classes = num_classes
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.chosen_divergence = chosen_divergence
    self.lr = lr
    self.save_hyperparameters()

    self.generator = Generator(image_size=self.img_size, hidden_dim=self.hidden_dim, z_dim=latent_dim)
    self.discriminator = Discriminator(image_size=self.img_size, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
    
    self.validation_z = torch.randn(self.batch_size, self.latent_dim)
    self.g_criterion = GeneratorLoss(chosen_divergence = self.chosen_divergence)
    self.d_criterion = DiscriminatorLoss(chosen_divergence = self.chosen_divergence)

  def forward(self, z):
    return self.generator.forward(z)

  def training_step(self, batch, batch_idx, optimizer_idx):
    imgs, _ = batch

    #create sample generated images
    z = torch.randn(self.batch_size, self.hparams.latent_dim).type_as(imgs)

    #train generator
    if optimizer_idx == 0:
      generated_images = self.forward(z)
      discriminator_output_fake = self.discriminator.forward(generated_images)

      loss_G = self.g_criterion.compute_loss(discriminator_output_fake)

      self.log("train/Q_loss", loss_G, on_epoch=True)

      output = OrderedDict(
          {
          "loss": loss_G,
          "log": loss_G,
          }
      )
      return output

    #Train discriminator
    elif optimizer_idx == 1:
      
      #loss on real images
      discriminator_output_real_imgs = self.discriminator.forward(imgs)
      loss_real_imgs = self.d_criterion.compute_loss(discriminator_output_real_imgs)

      #loss on fake images
      generated_images = self.forward(z)
      discriminator_generated_imgs_output = self.discriminator(generated_images)
      loss_generated_imgs = -self.g_criterion.compute_loss(discriminator_generated_imgs_output)

      total_loss_D = -(loss_real_imgs + loss_generated_imgs)

      self.log("train/V_loss", total_loss_D, on_epoch=True)

      output = OrderedDict(
          {
          "loss": total_loss_D,
          "log": total_loss_D
          }
        )
      
  def configure_optimizers(self):
    optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
    optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)

    return [optimizer_G, optimizer_D]




    