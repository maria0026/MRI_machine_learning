import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.values.astype(np.float32))
        self.y = torch.from_numpy(y.values.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
    
class DataRNN(Dataset):
    def __init__(self, X, y, sequence_length):
        # Reshape X to include the sequence dimension if necessary
        if len(X.shape) == 2:  # Check if X is 2D and needs a third dimension
            input_size = X.shape[1]
            self.X = torch.from_numpy(X.values.astype(np.float32)).reshape(-1, sequence_length, input_size)
        else:
            self.X = torch.from_numpy(X.values.astype(np.float32))
        
        self.y = torch.from_numpy(y.values.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

def load_fnn_data(X, y, batch_size):
    train_data = Data(X, y)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    return train_dataloader

def load_rnn_data(X, y, batch_size):
    train_data = DataRNN(X, y, sequence_length=1)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    return train_dataloader

#to na razie nieużywane
class DataRNN_binary(Dataset):
    def __init__(self, X, y, sequence_length=1):
        # Reshape X to include the sequence dimension if necessary
        if len(X.shape) == 2:  # Check if X is 2D and needs a third dimension
            input_size = X.shape[1]
            self.X = torch.from_numpy(X.values.astype(np.float32)).reshape(-1, sequence_length, input_size)
        else:
            self.X = torch.from_numpy(X.values.astype(np.float32))
        
        self.y = torch.from_numpy(y.values.astype(np.int64))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len