import torch.nn as nn

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
    def __init__(self, layer_shapes):
      super(FF, self).__init__()
      # Put configuration layer shapes into list 
      self.layer_shapes = list(eval(layer_shapes))
      # Create linear layers 
      self.linears = nn.ModuleList([nn.Linear(inp, out, bias=False) for inp, out in self.layer_shapes])
    
    def forward(self, x):
      for i, layer in enumerate(self.linears):
        x = self.linears[i](x)
        # If last layer, do not pass through activation function, else do so
        if layer != self.linears[-1]:
          x = nn.functional.leaky_relu(x)
      
      return x