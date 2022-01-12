from numpy.lib.type_check import real
from src.models.modules.walter_work import *
from pytorch_lightning import LightningModule
import torch
from torchmetrics import Accuracy
from src.models.modules.loss_modules import VLOSS, QLOSS, Wasserstein_QLOSS, Wasserstein_VLOSS, REGLOSS
from src.models.modules.Q_network import Q_CNN, Q_DCGAN, Q_DCGAN_128
from src.models.modules.V_network import V_CNN, V_DCGAN, V_DCGAN_128


class WalterGAN(LightningModule):
    def __init__(
        self,
        div: str = "JSD",
        backbone:str = "DCGAN",
        bsize: int = 128,
        nc: int = 1,
        nz: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        height: int = 4,
        width: int = 4,
        lr: float = 0.00002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        use_noise: bool = False,
        noise_bandwith: float = 0.01,
        noise_annealing: float = 1.,
        c: float = 0.01,
        use_disc_reg: bool = False,
        reg_gama: float = 0.1,
        reg_annealing: float = 1.,
        ):
                
        super().__init__()
        self.div = div
        self.backbone = backbone,
        self.bsize = bsize
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
        
        if self.backbone == 'DCGAN':
            Q_net = Q_DCGAN(self.nz, self.ngf, self.nc).to(self.device)
            V_net = V_DCGAN(self.nc, self.ndf).to(self.device)
            
        elif self.backbone == 'CNN':
            Q_net = Q_CNN(self.nz, self.ngf, self.nc, height, width).to(self.device)
            V_net = V_CNN(self.nc, self.ndf, height, width).to(self.device)
            
        elif self.backbone == 'DCGAN_128':
            Q_net = Q_DCGAN_128(self.nz, self.ngf, self.nc).to(self.device)
            V_net = V_DCGAN_128(self.nc, self.ndf).to(self.device)
            
        else:
            Q_net = Q_CNN(self.nz, self.ngf, self.nc, height, width).to(self.device)
            V_net = V_CNN(self.nc, self.ndf, height, width).to(self.device)

        self.generator = Q_net
        self.discriminator = V_net
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        if self.div == 'Wasserstein':
            Q_criterion = Wasserstein_QLOSS(self.div)
            V_criterion = Wasserstein_VLOSS(self.div)
        else:
            Q_criterion = QLOSS(self.div)
            V_criterion = VLOSS(self.div)
        
        self.Q_criterion = Q_criterion
        self.V_criterion = V_criterion
            
        if self.use_disc_reg:
            reg_criterion = REGLOSS(self.div)
            self.reg_criterion = reg_criterion
        
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.d_accuracy_on_generated_instances = Accuracy(num_classes=1)

    def forward(self, z):
        return self.generator.forward(z)

    #def on_train_start(self) -> None:
    #    self.generator.apply(weights_init)
    #    self.discriminator.apply(weights_init)

    def training_step(self, batch, batch_idx):
        data, _ = batch
    
        z = self.sample_z().type_as(data)
        print(f"[INFO]: z example: {z[0]}")
        fake_data = self.generator.forward(z)
        print(f"[INFO]: fake_data example: {fake_data[0]}")

        if self.use_noise:
            annealed_bandwidth = self.noise_bandwith*(self.noise_annealing**self.current_epoch)
            noise_term = torch.randn(data.size()).to(self.device) * annealed_bandwidth
            input_data = data + noise_term
            noise_term = torch.randn(fake_data.size()).to(self.device) * annealed_bandwidth
            input_fake = fake_data + noise_term
        else:
            input_data = data
            input_fake = fake_data

        g_opt, d_opt = self.optimizers()

        ## TRAIN DISCRIMINATOR ##
        # Discriminator output on real instances
        v = self.discriminator(input_data)
        print(f"[INFO]: v output example: {v[:10]}")
        loss_real = -self.V_criterion(v)
        print(f"[INFO]: loss real: {loss_real}")
        #loss_real.backward(retain_graph=True)
            
        # Discriminator output on fake instances
        v_fake = self.discriminator(input_fake)
        print(f"[INFO] v_fake output example: {v_fake[:10]}")
        loss_fake = -self.Q_criterion(v_fake)
        print(f"[INFO] loss fake: {loss_fake}")
        #loss_fake.backward()#maximizes F

        loss_V = -(loss_real + loss_fake)

        if self.use_disc_reg:
            loss_V += self.reg_gama*self.reg_criterion(self.discriminator, input_fake.detach())

        self.log("train/V_loss", loss_V, on_epoch=True)
        d_opt.zero_grad()
        self.manual_backward(loss_V, retain_graph=True)
        d_opt.step()
        
        ## TRAIN GENERATOR ##
        # Discrimator output on fake instances
        v_fake = self.discriminator.forward(input_fake)
        loss_Q = -self.V_criterion(v_fake)
        self.log("train/Q_loss", loss_Q, on_epoch=True)
        
        g_opt.zero_grad()
        self.manual_backward(loss_Q)
        g_opt.step()


             
    def on_train_batch_end(self, outputs, batch, batch_idx, unused = 0) -> None:
        if self.div == 'Wasserstein':
            # Clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.c, self.c)

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.beta1, self.beta2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.beta1, self.beta2))
        return [optimizer_G, optimizer_D]

    def sample_z(self):
        #if self.backbone == "CNN":
        noise = torch.randn(self.bsize, self.nz)
        #else:
        #    noise = torch.randn(self.bsize, self.nz, 1, 1)

        return noise
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv')!=-1 or classname.find('Linear')!=-1:
            nn.init.normal_(m.weight.data,0.0,0.02)
        elif classname.find('BatchNorm')!=-1:
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)




    