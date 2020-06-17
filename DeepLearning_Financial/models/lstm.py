import pandas as pd 
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Sequence(nn.Module):
    def __init__(self, nb_features=1, hidden_size=100, nb_layers=5, dropout=0.5):
        super(Sequence, self).__init__()
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.nb_layers=nb_layers
        self.lstm = nn.LSTM(self.nb_features, self.hidden_size, self.nb_layers, dropout=dropout)
        self.lin = nn.Linear(self.hidden_size,1)
        h0 = torch.zeros(self.nb_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.nb_layers, 1, self.hidden_size)
        self.hidden_cell = (h0, c0)

    def forward(self, input):
        lstm_out, hn = self.lstm(input.view(len(input) ,1, -1), self.hidden_cell)
        #output = F.relu(self.lin(output))
        out = self.lin(lstm_out.view(len(input), -1))
        return out[-1]