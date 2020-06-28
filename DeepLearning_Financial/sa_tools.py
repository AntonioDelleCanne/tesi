## EXTERNAL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import yfinance
from pandas import Series
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, PredefinedSplit
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from IPython.display import display
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
import os
import random 
from sklearn.datasets import make_regression
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from torch.nn.modules.loss import MSELoss
import tensorflow as tf
from tensorflow import keras
from skorch.dataset import CVSplit
from skorch import callbacks
import pickle
from sklearn.model_selection import train_test_split
from functools import partial
import skorch
import pywt
from sklearn import preprocessing
import joblib


##INTERNAL
from models import Autoencoder, waveletSmooth, SequenceDouble, SequenceDoubleAtt, SequenceAtt
from utils import prepare_data_lstm, ExampleDataset, save_checkpoint, evaluate_lstm, backtest




def get_encoder(X, val=None, sa_hidden_size=10):
    X_train_f = X.astype(np.float32)
    if(val is not None):
        X_val_f = val.astype(np.float32)
    #Initialize the autoencoder
    sa_hidden_size= np.ceil(X.shape[1] / 2).astype(int) # Con tutte le features 10

    num_hidden_1 = sa_hidden_size
    num_hidden_2 = sa_hidden_size
    num_hidden_3 = sa_hidden_size
    num_hidden_4 = sa_hidden_size

    n_epoch1=15000 #10000
    n_epoch2 = 2000
    n_epoch3 = 600
    n_epoch4 = 500
    batch_size=20

    # ---- train using training data

    # The n==0 statement is done because we only want to initialize the network once and then keep training
    # as we move through time 

    auto1 = Autoencoder(X_train_f.shape[1], num_hidden_1)
    auto2 = Autoencoder(num_hidden_1, num_hidden_2)
    auto3 = Autoencoder(num_hidden_2, num_hidden_3)
    auto4 = Autoencoder(num_hidden_3, num_hidden_4)
    
    # Train the autoencoder 
    # switch to training mode
    auto1.train()      
    auto2.train()
    auto3.train()
    auto4.train()

    inputs = torch.from_numpy(X_train_f)
    val_in = torch.from_numpy(X_val_f)
    auto1.fit(X_train_f, X_val_f, n_epoch=n_epoch1, batch_size=batch_size)

    auto1_out = auto1.encoder(inputs).data.numpy()
    val1_out = auto1.encoder(val_in).data.numpy()
    auto2.fit(auto1_out, val1_out, n_epoch=n_epoch2, batch_size=batch_size)


    auto1_out = torch.from_numpy(auto1_out.astype(np.float32))
    auto2_out = auto2.encoder(auto1_out).data.numpy()
    val1_out = torch.from_numpy(val1_out.astype(np.float32))
    val2_out = auto2.encoder(val1_out).data.numpy()
    auto3.fit(auto2_out, val2_out, n_epoch=n_epoch3, batch_size=batch_size)


    auto2_out = torch.from_numpy(auto2_out.astype(np.float32))
    auto3_out = auto3.encoder(auto2_out).data.numpy()
    val2_out = torch.from_numpy(val2_out.astype(np.float32))
    val3_out = auto3.encoder(val2_out).data.numpy()
    auto4.fit(auto3_out, val3_out, n_epoch=n_epoch4, batch_size=batch_size)

    # Change to evaluation mode, in this mode the network behaves differently, e.g. dropout is switched off and so on
    auto1.eval()        
    auto2.eval()
    auto3.eval()
    auto4.eval()
    return [auto1, auto2, auto3, auto4]

def encode(feat_matrix, encoder):
    encoded = torch.from_numpy(feat_matrix)
    for auto in encoder:
        encoded = auto.encoder(encoded)
    return encoded.data.numpy()
    

