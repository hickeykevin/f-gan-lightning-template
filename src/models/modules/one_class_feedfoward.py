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