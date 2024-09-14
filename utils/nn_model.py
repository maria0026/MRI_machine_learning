import torch
from torch import nn
from torch.autograd import Variable

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim*2)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim*2, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.dropout = nn.Dropout(0.4)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim//2)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="relu")
        self.output_layer = nn.Linear(hidden_dim//2, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.layer_3(x))
        x = torch.nn.functional.sigmoid(self.output_layer(x))

        return x


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        #out = torch.relu(out)
        #sigmoid
        out = torch.sigmoid(out)
        return out