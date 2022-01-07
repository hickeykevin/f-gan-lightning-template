from src.models.modules.walter_work import *
from pytorch_lightning import LightningModule
import torch
from torchmetrics import Accuracy


class WalterGAN(LightningModule):
    def __init__(
        self,
        div: str = "JSD",
        nc: int = 1,
        nz: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        lr: float = 0.00002,
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
        self.params = PARAMS
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




    