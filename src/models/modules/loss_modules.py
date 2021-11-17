#main point
#decaring activation and conjugate functions
import torch
from torch import nn

class GeneratorLoss:
  def __init__(self, chosen_divergence: str):
    self.chosen_divergence = chosen_divergence

  def compute_loss(self, G_output):
    if self.chosen_divergence == "KLD":
      activation_G_output = G_output
      return -torch.mean(torch.exp(activation_G_output-1))

    elif self.chosen_divergence == "RKL":
      activation_G_output = -torch.exp(-G_output)
      return -torch.mean(-1 - torch.log(-activation_G_output))

    elif self.chosen_divergence == "CHI":
      activation_G_output = G_output
      return -torch.mean(0.25 * activation_G_output**2 + activation_G_output)

    elif self.chosen_divergence == "SQH":
      activation_G_output = 1-torch.exp(-G_output)
      return -torch.mean(G_output / (1. - G_output))

    elif self.chosen_divergence == "JSD":
      activation_G_output = torch.log(torch.tensor(2.)) - torch.log(1. + torch.exp(-G_output))
      return -torch.mean(-torch.log(2.0 - torch.exp(activation_G_output)))

    elif self.chosen_divergence == "GAN":
      activation_G_output = -torch.log(1. + torch.exp(-G_output))
      return -torch.mean(-torch.log(1 - torch.exp(activation_G_output)))


class DiscriminatorLoss:
  def __init__(self, chosen_divergence):
    self.chosen_divergence = chosen_divergence

  def compute_loss(self, D_output_on_real, D_output_on_fake):
    if self.chosen_divergence == "KLD":
      activation_output_real = D_output_on_real
      activation_output_fake = D_output_on_fake
      return torch.mean(activation_output_real) - torch.mean(torch.exp(activation_output_fake-1))
    
    elif self.chosen_divergence == "RKL":
      activation_output_real = -torch.exp(-D_output_on_real) 
      activation_output_fake = -torch.exp(-D_output_on_fake)
      return torch.mean(activation_output_real) - torch.mean(-1 - torch.log(-activation_output_fake))
    
    elif self.chosen_divergence == "CHI":
      activation_output_real = D_output_on_real 
      activation_output_fake = D_output_on_fake
      return torch.mean(activation_output_real) - torch.mean(0.25 * activation_output_fake**2 + activation_output_fake)

    elif self.chosen_divergence == "SQH":
      activation_output_real = 1-torch.exp(-D_output_on_real)
      activation_output_fake = 1-torch.exp(-D_output_on_fake)
      return torch.mean(activation_output_real) - torch.mean(activation_output_fake / (1. - activation_output_fake))
    
    elif self.chosen_divergence == "JSD":
      activation_output_real = torch.log(torch.tensor(2.)) - torch.log(1.0+torch.exp(-D_output_on_real))
      activation_output_fake = torch.log(torch.tensor(2.)) - torch.log(1.0+torch.exp(-D_output_on_fake))
      return torch.mean(activation_output_real) - torch.mean(-torch.log(2.0 - torch.exp(activation_output_fake)))
    
    elif self.chosen_divergence == "GAN":
      activation_output_real = -torch.log(1.0 + torch.exp(-D_output_on_real))
      activation_output_fake = -torch.log(1.0 + torch.exp(-D_output_on_fake))  
      return torch.mean(activation_output_real) - torch.mean(-torch.log(1 - torch.exp(activation_output_fake)))
    

ACTIVATIONS = {
    "KLD": lambda v: v,
    "RKL": lambda v: -torch.exp(-v),
    "CHI": lambda v: v,
    "SQH": lambda v: 1-torch.exp(-v),
    "JSD": lambda v: torch.log(torch.tensor(2.)) - torch.log(1.0+torch.exp(-v)),
    "GAN": lambda v: -torch.log(1.0 + torch.exp(-v)), 
}

CONJUGATES = {
    "KLD": lambda t: torch.exp(t-1),
    "RKL": lambda t: -1 - torch.log(-t),
    "CHI": lambda t: 0.25 * t**2 + t,
    "SQH": lambda t: t / (1. - t),
    "JSD": lambda t: -torch.log(2.0 - torch.exp(t)),
    "GAN": lambda t: -torch.log(1.0 - torch.exp(t)) 
    }

class Q_loss(nn.Module):
  def __init__(self, chosen_divergence):
    super(Q_loss, self).__init__()
    self.conjugates = CONJUGATES
    self.activations = ACTIVATIONS
    self.chosen_divergence = chosen_divergence

  def forward(self, v):
    activation_output = self.activations[self.chosen_divergence](v)
    return torch.mean(-self.conjugates[self.chosen_divergence](activation_output))

class V_loss(nn.Module):
  def __init__(self, chosen_divergence):
    super(V_loss, self).__init__()
    self.activations = ACTIVATIONS
    self.chosen_divergence = chosen_divergence

  def forward(self, v):
    return torch.mean(self.activations[self.chosen_divergence](v))
