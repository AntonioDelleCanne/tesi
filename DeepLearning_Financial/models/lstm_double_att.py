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
    def __init__(self, nb_features=1, hidden_size=100, dropout=0.5):
        super(SequenceDoubleAtt, self).__init__()
        nb_layers=1
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.nb_layers=nb_layers
        self.lstm1 = nn.LSTM(self.nb_features, self.hidden_size, self.nb_layers, dropout=0)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size*2, self.nb_layers, dropout=0)
        self.lin = nn.Linear(self.hidden_size*2,1)
        h01 = torch.zeros(self.nb_layers, 1, self.hidden_size)#TODO fix
        c01 = torch.zeros(self.nb_layers, 1, self.hidden_size)
        self.hidden_cell1 = (h01, c01)
        h02 = torch.zeros(self.nb_layers, 1, self.hidden_size*2)#TODO adatta per nuova struttura
        c02 = torch.zeros(self.nb_layers, 1, self.hidden_size*2)
        self.hidden_cell2 = (h02, c02)
        self.softmax = nn.Softmax(0)
        #aggiungi softmax function

    def forward(self, input):
        lstm1_out, hn1 = self.lstm1(input.view(len(input) ,1, -1), self.hidden_cell1)
        lstm2_out, hn2 = self.lstm2(lstm1_out.view(len(input) ,1, -1), self.hidden_cell2)
        #output = F.relu(self.lin(output))
        out = self.lin(lstm2_out.view(len(input), -1))
        w = self.softmax(out)
        res = torch.mean(out*w, axis=0)
        return res
    
    
    
    