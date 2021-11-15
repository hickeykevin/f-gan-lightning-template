import torch.nn as nn
from itertools import zip_longest

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, rep_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=None) 
        # Non-linearity
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, rep_dim, bias=None)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity  # NON-LINEAR
        out = self.relu(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out

class FF(nn.Module):
    def __init__(self, layer_shapes, use_batch_norm=True):
      super(FF, self).__init__()
      # Put configuration layer shapes into list 
      self.layer_shapes = list(eval(layer_shapes))
      self.use_batch_norm = use_batch_norm
      # Create linear, activation, and optional batch_norm layers layers 
      self.linears = nn.ModuleList([nn.Linear(inp, out, bias=False) for inp, out in self.layer_shapes])
      self.batch_norm_layers = nn.ModuleList(list(nn.BatchNorm1d(out) for _, out in self.layer_shapes[:-1]))
      self.activations = nn.ModuleList(list(nn.LeakyReLU() for _ in self.layer_shapes[:-1]))
      # Combine all layers into a list
      self.blocks = list(zip_longest(self.linears, self.batch_norm_layers, self.activations))

    def forward(self, x):
      for block in self.blocks:
       for layer in block:
         if layer != None:
          x = layer(x) 
      return x

class Network(nn.Module):
  def __init__(self, input_size, use_batch_norm=True, **kwargs):
    super(Network, self).__init__()
    self.use_batch_norm = use_batch_norm
    
    # Initialize list of layers; first is layer to receive data
    layers = [nn.Linear(input_size, layer_sizes[0][1], bias=False)]
    
    # Store specified layer sizes in list;
    # When performing hyperparameter tuning, 
    # If layer size = 0, then do not include that layer 
    layer_sizes = [x for x in list(kwargs.items()) if x[-1] != 0]
    for i, x in enumerate(layer_sizes):
      if i != len(layer_sizes)-1:
        layers.append(nn.Linear(x[-1], layer_sizes[i+1][-1], bias=False))
    
    # Create a block of Linear, Batch_Norm, LeakyRelu layers
    layers = nn.ModuleList(layers)
    activations = nn.ModuleList(list(nn.LeakyReLU() for _ in layer_sizes[:-1]))
    
    if self.use_batch_norm:
      batch_norm_layers = nn.ModuleList(list(nn.BatchNorm1d(out[-1]) for out in layer_sizes[:-1]))
      self.blocks = list(zip_longest(layers, batch_norm_layers, activations))
    else:
      self.blocks = list(zip_longest(layers, activations))


  def forward(self, x):
    for block in self.blocks:
      for layer in block:
        #last layer does not use activation of batch norm
        if layer != None:
          x = layer(x)
    return x