import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim*2)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim*2, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.dropout = nn.Dropout(0.5)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim//2)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="relu")
        self.output_layer = nn.Linear(hidden_dim//2, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.tanh(self.layer_2(x))
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.layer_3(x))
        x = torch.nn.functional.sigmoid(self.output_layer(x))

        return x
