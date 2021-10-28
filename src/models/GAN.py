from src.models.modules.Q_network import Q
from src.models.modules.V_network import V

from src.models.modules.loss_modules import ACTIVATIONS, CONJUGATES, Q_loss, V_loss
from pytorch_lightning import LightningModule

import torch
from collections import OrderedDict
from torchvision.utils import make_grid

class LitFGAN(LightningModule):
  def __init__(
      self,
      latent_dim: int = 100,
      img_shape = [1, 28, 28],
      num_classes: int = 10,
      lr: float = 0.0002,
      batch_size: int = 64,
      chosen_divergence: str = "KLD",
      activations: dict = ACTIVATIONS,
      conjugates: dict = CONJUGATES
              ):
    
    super().__init__()
    self.latent_dim = latent_dim
    self.img_shape = img_shape
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.chosen_divergence = chosen_divergence
    self.save_hyperparameters()

    self.Q = Q(self.latent_dim)
    self.V = V()
    self.validation_z = torch.randn(self.img_shape[0], self.latent_dim)

  def forward(self, z):
    return self.Q(z)

  def training_step(self, batch, batch_idx, optimizer_idx):
    imgs, _ = batch

    #create sample generated images
    z = torch.randn(imgs.shape[0], self.hparams.latent_dim).type_as(imgs)
    q_criterion = Q_loss(chosen_divergence = self.chosen_divergence)
    v_criterion = V_loss(chosen_divergence = self.chosen_divergence)

    #train Q
    if optimizer_idx == 0:
      generated_images = self(z)
      v_output_fake = self.V(generated_images)

      loss_Q = q_criterion(v_output_fake)

      self.log("train/Q_loss", loss_Q, on_epoch=True)

      output = OrderedDict(
          {
          "loss": loss_Q,
          "log": loss_Q,
          }
      )
      return output

    #Train V
    elif optimizer_idx == 1:
      
      #loss on real images
      v_real_imgs_output = self.V(imgs)
      loss_real_imgs = v_criterion(v_real_imgs_output)

      #loss on fake images
      generated_images = self.forward(z)
      v_generated_imgs_output = self.V(generated_images)
      loss_generated_imgs = -q_criterion(v_generated_imgs_output)

      total_loss_v = -(loss_real_imgs + loss_generated_imgs)

      self.log("train/V_loss", total_loss_v, on_epoch=True)


      output = OrderedDict(
          {
          "loss": total_loss_v,
          "log": total_loss_v
          }
        )
      
  def configure_optimizers(self):
    lr = self.hparams.lr

    optimizer_q = torch.optim.Adam(self.Q.parameters(), lr=lr)
    optimizer_v = torch.optim.Adam(self.V.parameters(), lr=lr)

    return [optimizer_q, optimizer_v]

  def training_epoch_end(self, outputs):
    pass
    #wandb = self.logger.experiment[0]
    #generated_images = make_grid(self(self.validation_z).to(device=pl_module.device)
    #generated_images = wandb.Image(generated_images)
    #wandb.add_images("generated_examples", generated_images_outputs)


    