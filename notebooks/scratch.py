#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch.distributions.multivariate_normal import MultivariateNormal
#%%
data_dir: str = "../data/"
train_one_class: bool = True
train_val_test_split = (55000, 5000, 10000)
batch_size: int = 64
num_workers: int = 0
pin_memory: bool = False
chosen_class = [7]
transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
#MNIST(data_dir, train=True, download=True)
#MNIST(data_dir, train=False, download=True)
trainset = MNIST(data_dir, train=True, transform=transforms)
testset = MNIST(data_dir, train=False, transform=transform)
            
# %%
chosen_class = [7, 5]
if train_one_class:
    indices = torch.isin(torch.tensor(trainset.targets)[..., None], torch.tensor([chosen_class])).any(-1).nonzero(as_tuple=True)[0]
torch.utils.data.Subset(trainset, indices)
# %%
noise_z = MultivariateNormal(torch.zeros(784, torch.eye(784))).sample()

# %%
