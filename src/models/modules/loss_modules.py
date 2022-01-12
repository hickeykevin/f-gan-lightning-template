import torch
from torch import nn

class QLOSS(nn.Module):
    def __init__(self,divergence):
        super(QLOSS,self).__init__()
        self.conjugate = Conjugate_f(divergence)
        self.activation = Activation_g(divergence)
    def forward(self,v):
        return torch.mean(-self.conjugate(self.activation(v)))

class VLOSS(nn.Module):
    def __init__(self,divergence):
        super(VLOSS,self).__init__()
        self.activation = Activation_g(divergence)
    def forward(self,v):
        return torch.mean(self.activation(v))

class REGLOSS(nn.Module):
    def __init__(self,divergence):
        super(REGLOSS,self).__init__()
        self.activation = Activation_g(divergence)
        self.f_double = Conjugate_double_prime(divergence)
        
    def forward(self, disc, fake_data):
        flat_data = fake_data.view(fake_data.shape[0], -1)
        T = combined_T(disc, self.activation, fake_data.shape)
        grad_T_logits = compute_delta_x(T, flat_data)
        grad_T_logits_norm = torch.norm(grad_T_logits, dim=1, keepdim=True)
        t = T(flat_data)
        f_dprime_t = self.f_double(t)
        disc_regularizer = grad_T_logits_norm*(f_dprime_t**2)
        disc_regularizer = torch.mean(disc_regularizer)
        return disc_regularizer

class Wasserstein_VLOSS(nn.Module):
    def __init__(self,divergence="Wasserstein"):
        super(Wasserstein_VLOSS,self).__init__()
    def forward(self,v):
        return torch.mean(v)

class Wasserstein_QLOSS(nn.Module):
    def __init__(self,divergence="Wasserstein"):
        super(Wasserstein_QLOSS,self).__init__()
    def forward(self,v):
        return -torch.mean(v)


def conv_bn_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    )

def tconv_bn_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    )

def tconv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

def conv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

def fc_layer(in_features,out_features):
    return nn.Linear(in_features,out_features)

def fc_bn_layer(in_features,out_features):
    return nn.Sequential(
        nn.Linear(in_features,out_features),
        nn.BatchNorm1d(out_features)
    )

def compute_delta_x(net, x):
    delta_x = torch.autograd.functional.jacobian(net, x, create_graph=True)
    delta_x = torch.diagonal(delta_x, dim1=0, dim2=2).permute(2, 0, 1)
    delta_x = delta_x.reshape(delta_x.shape[0], delta_x.shape[2])
    return delta_x


class Activation_g(nn.Module):
    def __init__(self,divergence):
        super(Activation_g,self).__init__()
        self.divergence =divergence
    def forward(self,v):
        divergence = self.divergence
        if divergence == "KLD":
            return v
        elif divergence == "RKL":
            return -torch.exp(-v)
        elif divergence == "CHI":
            return v
        elif divergence == "SQH":
            return 1-torch.exp(-v)
        elif divergence == "JSD":
            return torch.log(torch.tensor(2.))-torch.log(1.0+torch.exp(-v))
        elif divergence == "GAN":
            return -torch.log(1.0+torch.exp(-v)) # log sigmoid

class Conjugate_f(nn.Module):
    def __init__(self,divergence):
        super(Conjugate_f,self).__init__()
        self.divergence = divergence
    def forward(self,t):
        divergence= self.divergence
        if divergence == "KLD":
            return torch.exp(t-1)
        elif divergence == "RKL":
            return -1 -torch.log(-t)
        elif divergence == "CHI":
            return 0.25*t**2+t
        elif divergence == "SQH":
            return t/(torch.tensor(1.)-t)
        elif divergence == "JSD":
            return -torch.log(2.0-torch.exp(t))
        elif divergence == "GAN":
            return  -torch.log(1.0-torch.exp(t))

class Conjugate_double_prime(nn.Module):
    def __init__(self,divergence):
        super(Conjugate_double_prime,self).__init__()
        self.divergence = divergence
    def forward(self,v):
        divergence= self.divergence
        if divergence == "KLD":
            return torch.exp(v-1)
        elif divergence == "RKL":
            return 1./(v**2)
        elif divergence == "CHI":
            return 0.5*(v**0)
        elif divergence == "SQH":
            return (2*v)/((1-v)**3) + 2/((1-v)**2)
        elif divergence == "JSD":
            return 2*(torch.exp(v))/((2-torch.exp(v))**2)
        elif divergence == "GAN":
            return  (torch.exp(v))/((1-torch.exp(v))**2)

class combined_T(nn.Module):
    def __init__(self,disc, activation, original_shape):
        super(combined_T,self).__init__()
        self.disc = disc
        self.activation = activation
        self.original_shape = original_shape
    def forward(self,x):
        x = x.view(self.original_shape)
        out = self.activation(self.disc(x))
        return out.view(out.shape[0], 1)

class net_with_flat_input(nn.Module): 
    def __init__(self,net, original_shape):
        super(net_with_flat_input,self).__init__()
        self.net = net
        self.original_shape = original_shape
    def forward(self,x):
        x = x.view(self.original_shape)
        out = self.net(x)
        return out.view(out.shape[0], 1)





## OLDER CODE ##
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


class GeneratorLoss:
  def __init__(self, chosen_divergence: str):
    assert chosen_divergence in ["KLD", "RKL", "CHI", "SQH", "JSD", "GAN"]
    self.chosen_divergence = chosen_divergence

  def compute_loss(self, D_output):
    if self.chosen_divergence == "KLD":
      activation_D_output = D_output
      return -torch.mean(torch.exp(activation_D_output-1))

    elif self.chosen_divergence == "RKL":
      activation_D_output = -torch.exp(-D_output)
      return -torch.mean(-1 - torch.log(-activation_D_output))

    elif self.chosen_divergence == "CHI":
      activation_D_output = D_output
      return -torch.mean(0.25 * activation_D_output**2 + activation_D_output)

    elif self.chosen_divergence == "SQH":
      activation_D_output = 1-torch.exp(-D_output)
      return -torch.mean(D_output / (1. - D_output))

    elif self.chosen_divergence == "JSD":
      activation_D_output = torch.log(torch.tensor(2.)) - torch.log(1. + torch.exp(-D_output))
      return -torch.mean(-torch.log(2.0 - torch.exp(activation_D_output)))

    elif self.chosen_divergence == "GAN":
      activation_D_output = -torch.log(1. + torch.exp(-D_output))
      return -torch.mean(-torch.log(1 - torch.exp(activation_D_output)))


class DiscriminatorLoss:
  def __init__(self, chosen_divergence):
    assert chosen_divergence in ["KLD", "RKL", "CHI", "SQH", "JSD", "GAN"]
    self.chosen_divergence = chosen_divergence

  def compute_loss(self, D_output_on_real, D_output_on_fake):
    if self.chosen_divergence == "KLD":
      activation_output_real = D_output_on_real
      activation_output_fake = D_output_on_fake
      return -(torch.mean(activation_output_real) - torch.mean(torch.exp(activation_output_fake-1)))
    
    elif self.chosen_divergence == "RKL":
      activation_output_real = -torch.exp(-D_output_on_real) 
      activation_output_fake = -torch.exp(-D_output_on_fake)
      return -(torch.mean(activation_output_real) - torch.mean(-1 - torch.log(-activation_output_fake)))
    
    elif self.chosen_divergence == "CHI":
      activation_output_real = D_output_on_real 
      activation_output_fake = D_output_on_fake
      return -(torch.mean(activation_output_real) - torch.mean(0.25 * activation_output_fake**2 + activation_output_fake))

    elif self.chosen_divergence == "SQH":
      activation_output_real = 1-torch.exp(-D_output_on_real)
      activation_output_fake = 1-torch.exp(-D_output_on_fake)
      return -(torch.mean(activation_output_real) - torch.mean(activation_output_fake / (1. - activation_output_fake)))
    
    elif self.chosen_divergence == "JSD":
      activation_output_real = torch.log(torch.tensor(2.)) - torch.log(1.0+torch.exp(-D_output_on_real))
      activation_output_fake = torch.log(torch.tensor(2.)) - torch.log(1.0+torch.exp(-D_output_on_fake))
      return -(torch.mean(activation_output_real) - torch.mean(-torch.log(2.0 - torch.exp(activation_output_fake))))
    
    elif self.chosen_divergence == "GAN":
      activation_output_real = -torch.log(1.0 + torch.exp(-D_output_on_real))
      activation_output_fake = -torch.log(1.0 + torch.exp(-D_output_on_fake))  
      return -(torch.mean(activation_output_real) - torch.mean(-torch.log(1 - torch.exp(activation_output_fake))))
