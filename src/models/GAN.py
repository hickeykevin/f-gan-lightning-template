from src.models.modules.Q_network import Generator
from src.models.modules.V_network import Discriminator

from src.models.modules.loss_modules import DiscriminatorLoss, GeneratorLoss
from pytorch_lightning import LightningModule

import torch
from torch.distributions.uniform import Uniform
from torchmetrics import Accuracy
from torchvision.utils import make_grid


class LitFGAN(LightningModule):
  def __init__(
      self,
      latent_dim: int = 100,
      img_size = 784,
      hidden_dim = 512,
      output_dim = 1,
      lr: float = 0.0002,
      chosen_divergence: str = "KLD",
      batch_size=64
              ):
    
    super().__init__()
    self.latent_dim = latent_dim
    self.img_size = img_size
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.lr = lr
    self.chosen_divergence = chosen_divergence
    self.batch_size = batch_size
    self.save_hyperparameters()

    self.generator = Generator(image_size=self.img_size, hidden_dim=self.hidden_dim, z_dim=self.latent_dim).to(self.device)
    self.discriminator = Discriminator(image_size=self.img_size, hidden_dim=self.hidden_dim, output_dim=self.output_dim).to(self.device)
    
    self.validation_z = torch.randn(self.batch_size, self.latent_dim)
    self.g_criterion = GeneratorLoss(chosen_divergence = self.chosen_divergence)
    self.d_criterion = DiscriminatorLoss(chosen_divergence = self.chosen_divergence)
    self.d_accuracy_on_generated_instances = Accuracy(num_classes=1)

  def forward(self, z):
    return self.generator.forward(z)

  def training_step(self, batch, batch_idx, optimizer_idx):
    imgs, _ = batch

    # Train generator
    if optimizer_idx == 0:
      # Create sample of noise
      z = Uniform(-1, 1).sample([self.batch_size, self.hparams.latent_dim]).type_as(imgs)
      generated_images = self.forward(z)

      # Discriminator output on generated instances
      discriminator_output_generated_imgs = self.discriminator.forward(generated_images)

      # Loss calculation for Generator
      loss_G = self.g_criterion.compute_loss(discriminator_output_generated_imgs)

      self.log("train/G_loss", loss_G, on_epoch=True)

      output = {"loss": loss_G}
      return output

    # Train discriminator
    elif optimizer_idx == 1:
      # Discriminator output on real instances
      discriminator_output_real_imgs = self.discriminator.forward(imgs.view(self.batch_size, -1))
      
      # Discriminator output on fake instances
      # Create sample of noise
      z = Uniform(-1, 1).sample([self.batch_size, self.hparams.latent_dim]).type_as(imgs)
      generated_images = self.forward(z)
      discriminator_output_generated_imgs = self.discriminator.forward(generated_images)
      
      # Loss calculation for discriminator 
      loss_D = self.d_criterion.compute_loss(discriminator_output_real_imgs, discriminator_output_generated_imgs)

      self.d_accuracy_on_generated_instances(torch.sigmoid(discriminator_output_generated_imgs), torch.zeros((imgs.size()[0], 1), dtype=torch.int8))

      self.log("train/D_loss", loss_D, on_epoch=True)
      self.log("train/D_accuracy_generated_instances", self.d_accuracy_on_generated_instances, on_epoch=True)
      
      output = {"loss": loss_D}
      return output
        
      
  def configure_optimizers(self):
    optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
    optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)

    return [optimizer_G, optimizer_D]




    