import pandas as pd 
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class SequenceAtt(nn.Module):
    def __init__(self, nb_features=1, nb_layers=1, hidden_size=100, dropout=0):
        super(SequenceAtt, self).__init__()
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.nb_layers=nb_layers
        self.lstm1 = nn.LSTM(self.nb_features, self.hidden_size, self.nb_layers, dropout=0, batch_first=True)
        self.lin = nn.Linear(self.hidden_size,1)
        self.lin_out = nn.Linear(self.hidden_size,1)
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        h01 = torch.zeros(self.nb_layers, input.size()[0], self.hidden_size)
        c01 = torch.zeros(self.nb_layers, input.size()[0], self.hidden_size)
        hidden_cell1 = (h01, c01)
        lstm1_out, hn1 = self.lstm1(input, hidden_cell1)
        e = self.tanh(self.lin(lstm1_out.reshape(-1, lstm1_out.size()[-1]))).reshape(input.size()[0], input.size()[1], -1)
        w = self.softmax(e)
        att_out = torch.mean(lstm1_out*w, axis=1)
        res = self.lin_out(att_out)
        return res
    
    
    
    