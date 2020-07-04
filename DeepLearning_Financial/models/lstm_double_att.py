import pandas as pd 
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class SequenceDoubleAtt(nn.Module):
    def __init__(self, nb_features=1, hidden_size=100, dropout=0):
        super(SequenceDoubleAtt, self).__init__()
        nb_layers=1
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.nb_layers=nb_layers
        self.lstm1 = nn.LSTM(self.nb_features, self.hidden_size*2, self.nb_layers, dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size, self.nb_layers, dropout=dropout, batch_first=True)
        self.lin = nn.Linear(self.hidden_size,1)
        self.lin_out = nn.Linear(self.hidden_size,1)
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        h01 = torch.zeros(self.nb_layers, input.size()[0], self.hidden_size*2)
        c01 = torch.zeros(self.nb_layers, input.size()[0], self.hidden_size*2)
        hidden_cell1 = (h01, c01)
        h02 = torch.zeros(self.nb_layers, input.size()[0], self.hidden_size)
        c02 = torch.zeros(self.nb_layers, input.size()[0], self.hidden_size)
        hidden_cell2 = (h02, c02)
        lstm1_out, hn1 = self.lstm1(input, hidden_cell1)
        lstm2_out, hn2 = self.lstm2(lstm1_out, hidden_cell2)
        e = self.tanh(self.lin(lstm2_out.reshape(-1, lstm2_out.size()[-1]))).reshape(input.size()[0], input.size()[1], -1)
        w = self.softmax(e)
        att_out = torch.mean(lstm2_out*w, axis=1)
        res = self.lin_out(att_out)
        return res
    
    
    
    