import torch
import torch.nn as nn

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