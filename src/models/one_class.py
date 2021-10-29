#kevin's deep ocsvm translation into pytorch lightinng module
import pytorch_lightning as pl
import torch
from src.models.modules.one_class_feedforward import FeedforwardNeuralNetModel
from torchmetrics import AUROC


class LitDeepOCSVM(pl.LightningModule):
  def __init__(self, input_dim, hidden_dim, rep_dim, l2_weight = 0.01, lr = 1e-2):
    super().__init__()

    #keep same initialize parameters as walter
    #remember, these are for the feed-forward class 
    self.input_dim = input_dim
    self.hidden_dim  = hidden_dim
    self.rep_dim  = rep_dim
    self.num_classes = num_classes
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
    self.center_defined = False
    self.auroc = AUROC(num_classes=1)
    #self.conustion_matrix = ConfusionMatrix(num_classes=self.num_classes, normalize=True)

  
  def forward(self, x):
      #same as feed forward network's forward method
      x = self.model.forward(x)
      return x
  
  def define_center(self):
    pass

  
  def training_step(self, batch, batch_idx):
      X, y = batch
      X = X.reshape(-1, X.size()[-2] * X.size()[-1])
      #y = torch.ones(X.size()[0])

      f_X = self.forward(X)

      #walter's center defining logic
      if not self.center_defined:
        #print('Defining C')
        self.center_vec = torch.mean(f_X.detach(), dim=0)
        #print(self.center_vec.shape)
      self.center = self.center_vec.repeat(1,X.shape[0]) 
      self.center = self.center.view(X.shape[0], -1)
      self.center_defined = True

      #loss calculation, same procedure as walter's implementation
      loss = torch.norm(f_X - self.center)
      l2_reg = torch.tensor(0.).to(self.device)
      for param in self.model.parameters():
        l2_reg += torch.norm(param)
      loss += self.l2_weight * l2_reg
      #can be converted into a torchmetric class; will look into it


      #log the epoch level training loss to the progress bar
      self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
      self.log("train/auroc", self.auroc(f_X, y), on_step=False, on_epoch=True)
      
      return {"loss": loss}

  
  def validation_step(self, batch, batch_idx):
      X, y = batch
      X = X.reshape(-1, X.size()[-2] * X.size()[-1])
      #y = torch.ones(X.size()[0])
      f_X = self.forward(X)

      #walter's center defining logic
      if not self.center_defined:
        #print('Defining C')
        self.center_vec = torch.mean(f_X.detach(), dim=0)
        #print(self.center_vec.shape)
      self.center = self.center_vec.repeat(1,X.shape[0]) 
      self.center = self.center.view(X.shape[0], -1)
      self.center_defined = True

      #loss calculation; same as training step
      #MAKE THIS A SEPERATE FUNCTION/TORCHMETRIC?
      loss = torch.norm(f_X - self.center)
      l2_reg = torch.tensor(0.).to(self.device)
      for param in self.model.parameters():
        l2_reg += torch.norm(param)
      loss += self.l2_weight * l2_reg

      self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
      self.log("val/auroc", self.auroc(f_X, y), on_step=False, on_epoch=True)

      return {
        "loss": loss,
        "preds": f_X,
        "targets": y}


  #walter's optimizer implementation, conveted into PL form
  def configure_optimizers(self):
      optimizer =  torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
      return ([optimizer], [scheduler])