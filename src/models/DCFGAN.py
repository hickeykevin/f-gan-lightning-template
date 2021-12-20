from src.models.modules.dcgan import Generator, Discriminator
from src.models.modules.loss_modules import DiscriminatorLoss, GeneratorLoss
from pytorch_lightning import LightningModule
import torch
from torch.distributions.uniform import Uniform
from torchmetrics import Accuracy


class LitDCFGAN(LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        lr: float = 0.0002,
        chosen_divergence: str = "KLD",
        batch_size=64,
        adam_beta_one: float = 0.5,
        num_channels: int = 3
                ):
                
        super().__init__()
        self.latent_dim = latent_dim
        self.lr = lr
        self.chosen_divergence = chosen_divergence
        self.batch_size = batch_size
        self.adam_beta_one = adam_beta_one
        self.num_channels = num_channels
        self.save_hyperparameters()

        self.generator = Generator(self.num_channels, self.latent_dim).to(self.device)
        self.discriminator = Discriminator(self.num_channels, self.latent_dim).to(self.device)

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
            z = self.sample_z(self.batch_size).type_as(imgs)
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
            discriminator_output_real_imgs = self.discriminator.forward(imgs)

            # Discriminator output on fake instances
            z = self.sample_z(self.batch_size).type_as(imgs)
            generated_images = self.forward(z)
            discriminator_output_generated_imgs = self.discriminator.forward(generated_images)

            # Loss calculation for discriminator 
            loss_D = self.d_criterion.compute_loss(discriminator_output_real_imgs, discriminator_output_generated_imgs)

            # Log Metrics
            self.log("train/D_loss", loss_D, on_epoch=True)
            self.d_accuracy_on_generated_instances(torch.sigmoid(discriminator_output_generated_imgs.view(-1, 1)), torch.zeros((imgs.size()[0], 1), dtype=torch.int8))
            self.log("train/D_accuracy_generated_instances", self.d_accuracy_on_generated_instances, on_epoch=True)

            output = {"loss": loss_D}
            return output

        
    def sample_z(self, n):
        z = torch.randn(n, self.latent_dim, 1, 1)
        return z

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.adam_beta_one, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.adam_beta_one, 0.999))

        return [optimizer_G, optimizer_D]




    