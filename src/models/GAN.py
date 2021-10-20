from src.models.modules.Q_network import Q
from src.models.modules.V_network import V

from src.models.modules.loss_modules import ACTIVATIONS, CONJUGATES, Q_loss, V_loss
from pytorch_lightning import LightningModule

class LitFGAN(LightningModule):
  def __init__(
      self,
      latent_dim,
      img_shape,
      num_classes,
      lr,
      batch_size,
      chosen_divergence,
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

    self.Q = Q(self.latent_dim, self.img_shape, self.num_classes)
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

      #generated_images_labels = torch.ones(imgs.size[0], 1).type_as(images)
      loss_Q = q_criterion(generated_images)

      self.log("train/Q_loss", loss_Q, on_epoch=True)

      output = OrderedDict(
          {
          "loss": loss_Q,
          "log": loss_Q,
          "generated_examples": generated_images
          }
      )
      return output

    #Train V
    elif optimizer_idx == 1:
      #loss on real images
      v_real_imgs_output = self.V(imgs)
      loss_real_imgs = v_criterion(v_real_imgs_output)

      #loss on fake images
      generated_images = self(z)
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
    tensorboard = self.logger.experiment
    generated_images_outputs = outputs[0]["generated_examples"][:4]
    tensorboard.add_images("generated_examples", generated_images_outputs)



    