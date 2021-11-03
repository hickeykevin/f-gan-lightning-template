#kevin's deep ocsvm translation into pytorch lightinng module
import pytorch_lightning as pl
import torch
from src.models.modules.one_class_feedforward import FeedforwardNeuralNetModel
from torchmetrics import AUROC, F1


class LitDeepOCSVM(pl.LightningModule):
  def __init__(self, input_dim, hidden_dim, rep_dim, l2_weight = 0.01, lr = 1e-2):
    super().__init__()

    #keep same initialize parameters as walter
    #remember, these are for the feed-forward class 
    self.input_dim = input_dim
    self.hidden_dim  = hidden_dim
    self.rep_dim  = rep_dim
    self.l2_weight = l2_weight

    #`batch_size` parameter moved to Lightning DataModule
    #self.batch_size = batch_size

    #`device` not needed
    #self.device  = device
    self.lr  = lr

    #`num_epochs` is handled by PL trainer object
    #self.num_epochs  = num_epochs

    #`verbose` not needed
    #self.verbose  = verbose
    
    self.model = FeedforwardNeuralNetModel(input_dim = self.input_dim, hidden_dim = self.hidden_dim,
                                            rep_dim = self.rep_dim).to(self.device)
    self.auroc = AUROC(num_classes=2, pos_label=1)
  
  def forward(self, x):
      #same as feed forward network's forward method
      x = self.model.forward(x)
      return x

  def on_train_start(self):
    #define the center based on 1 pass through the data
    #
    batch = next(iter(self.trainer.train_dataloader()))
    X, y = batch
    X = X.reshape(-1, X.size()[-2] * X.size()[-1])
    f_X = self.forward(X)
    self.center_vec = torch.mean(f_X.detach(), dim=0)
    self.center = self.center_vec.repeat(1,X.shape[0]) 
    self.center = self.center.view(X.shape[0], -1)


  
  def training_step(self, batch, batch_idx):
    #remember, these instances will be filtered to self.trainer.datamodule.chosen_class
    #handled by the dataloader
      X, y = batch
      X = X.reshape(-1, X.size()[-2] * X.size()[-1])

      f_X = self.forward(X)

      #loss calculation, same procedure as walter's implementation
      loss = torch.norm(f_X - self.center)
      l2_reg = torch.tensor(0.).to(self.device)
      for param in self.model.parameters():
        l2_reg += torch.norm(param)
      loss += self.l2_weight * l2_reg
      #can be converted into a torchmetric class; will look into it


      #log the epoch level training loss to the progress bar
      self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
      
      return {"loss": loss}

  
  def validation_step(self, batch, batch_idx):
      #any image not pertaining to self.chosen_class has label of 0
      X, y = batch
      y[y != self.trainer.datamodule.chosen_class] = 0
      y[y == self.trainer.datamodule.chosen_class] = 1
      X = X.reshape(-1, X.size()[-2] * X.size()[-1])
      f_X = self.forward(X)
      
      #loss calculation; same as training step
      #MAKE THIS A SEPERATE FUNCTION/TORCHMETRIC?
      loss = torch.norm(f_X - self.center)
      l2_reg = torch.tensor(0.).to(self.device)
      for param in self.model.parameters():
        l2_reg += torch.norm(param)
      loss += self.l2_weight * l2_reg

      #whether instances are close to center or not
      score = -torch.norm((f_X - self.center), dim=1)

      self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
      self.log("val/auroc", self.auroc(score, y), on_step=True, on_epoch=True)
      #self.log("val/f1", self.f1(f_X.reshape(-1, 1), y.reshape(-1, 1)), on_step=True, on_epoch=True)
      
      return {"loss": loss}

  #walter's optimizer implementation, conveted into PL form
  def configure_optimizers(self):
      optimizer =  torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
      return ([optimizer], [scheduler])