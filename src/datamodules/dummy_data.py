#walter's pytorch dataset class
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.datasets import make_classification
import torch
class UnlabeledDataset(Dataset):
    def __init__(self, X):
        self.data = X
          
    def __getitem__(self, index):
        x = self.data[index]
        return x
      
    def __len__(self):
        return len(self.data)

#kevin's lightning data module class
#is dummy data, so results don't mean anything for this datamodule
#remember, will be fed with deep ocsvm trainer
class DummyLitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
      super().__init__()
      self.batch_size  = batch_size


    def prepare_data(self):
      #method not needed; just here for completeness
      pass
      
    def setup(self, stage):
      #could move each of make_classification args to config file too...
      X, y = make_classification(n_samples = 10000, n_features=7, n_informative=4, n_classes=1)
      self.X = torch.from_numpy(X).float()
      self.train_dataset = UnlabeledDataset(self.X)

    def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size)