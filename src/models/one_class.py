#kevin's deep ocsvm translation into pytorch lightinng module
import pytorch_lightning as pl
import torch
import torchmetrics
from src.models.modules.one_class_feedforward import FeedforwardNeuralNetModel
from torchmetrics import AUROC, PrecisionRecallCurve, Precision, Recall


class LitDeepOCSVM(pl.LightningModule):
  def __init__(
    self, 
    input_dim, 
    hidden_dim, 
    rep_dim, 
    #layer_shapes,
    #input_size,
    l2_weight = 0.01, 
    lr = 1e-2, 
    **kwargs):
    super().__init__()

    #arguments for the feed-forward class 
    self.input_dim = input_dim
    self.hidden_dim  = hidden_dim
    self.rep_dim  = rep_dim
    self.l2_weight = l2_weight
    self.lr  = lr
    self.model = FeedforwardNeuralNetModel(input_dim = self.input_dim, hidden_dim = self.hidden_dim,
                                            rep_dim = self.rep_dim).to(self.device)
    #self.model = Network(self.input_dim, kwargs, use_batch_norm=True)
    self.auroc = AUROC(num_classes=2, pos_label=1)
    self.pr_curve = PrecisionRecallCurve()
  
  def forward(self, x):
      #return anomoly score for a given instance
      x = self.model.forward(x)
      score = torch.norm(x - self.center, dim=1)**2
      return score

  def on_train_start(self):
      #define the center based on 1 pass through the data
      #get self.model output on batch of data from the datamodule
      batch = next(iter(self.trainer.train_dataloader))
      X, _ = batch
      X = X.reshape(-1, X.size()[-2] * X.size()[-1]).to(self.device)
      f_X = self.model.forward(X)

      #define the center
      self.center_vec = torch.mean(f_X.detach(), dim=0)
      self.center = self.center_vec.repeat(1,X.shape[0]) 
      self.center = self.center.view(X.shape[0], -1)


  def training_step(self, batch, batch_idx):
      #remember, these instances will be filtered to be equal to 
      #self.trainer.datamodule.chosen_class, handled by the dataloader
      X, y = batch
      X = X.reshape(-1, X.size()[-2] * X.size()[-1])
      f_X = self.model.forward(X)

      loss = self.loss_function(f_X)

      #log the epoch level training
      self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
      
      return {"loss": loss}

  
  def validation_step(self, batch, batch_idx):
      #any image not pertaining to self.chosen_class has label of 0
      X, y = batch

      #transform labels of targets
      if self.trainer.datamodule.chosen_class == 0 or self.trainer.datamodule.chosen_class == 1:
          y[y != self.trainer.datamodule.chosen_class] = -1
          y[y == self.trainer.datamodule.chosen_class] = 1
          y[y == -1] = 0
      else:
          y[y != self.trainer.datamodule.chosen_class] = 0
          y[y == self.trainer.datamodule.chosen_class] = 1
     
      X = X.reshape(-1, X.size()[-2] * X.size()[-1])
      f_X = self.model.forward(X)

      loss = self.loss_function(f_X)

      #produce score, indicating whether instances are close to center or not,
      #make negative for auroc score to give meaningful representation
      score = -(torch.norm(f_X - self.center, dim=1)**2)
      self.auroc(score, y)

      #log epoch level loss and auroc scores
      self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
      self.log("val/auroc", self.auroc, on_step=True, on_epoch=True)
      
      return {"loss": loss, "score": score, "targets": y}



  def loss_function(self, f_X):
      #take mean of squared difference of outputs and calculated center
      loss = torch.mean(torch.sum((f_X - self.center)**2, dim=1))

      #calculate regularizer parameter 
      l2_reg = torch.tensor(0.).to(self.device)
      for param in self.model.parameters():
        l2_reg += torch.sum(torch.linalg.norm(param)**2)

      #combine the two components to get total loss 
      loss += self.l2_weight * l2_reg

      return loss

  


  #walter's optimizer implementation, conveted into PL form
  def configure_optimizers(self):
      optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
      return ([optimizer], [scheduler])