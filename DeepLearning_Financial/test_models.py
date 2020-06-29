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


def get_model_names():
    return ['lstm_sa', 'lstm_att', 'lstm_moro', 'lstm_sa_att']

def get_model(model_name, save_name, train_split=None):
    model = None
    feature_set = None
    
    cb =[
        callbacks.EpochScoring('neg_mean_absolute_error', lower_is_better=False),
        callbacks.EpochScoring('r2', lower_is_better=False),
        callbacks.Checkpoint(monitor='valid_loss_best', f_pickle=save_name)        
    ]
    
    if(model_name is 'lstm_moro'):
        feature_set = 'open'
        batch_size = 20
        model = NeuralNetRegressor(
            module=SequenceDouble,
            optimizer=optim.Adam,
            batch_size=batch_size,
            max_epochs = 1000, # trovato empiricamente
            train_split=train_split,
            callbacks=cb,

            module__nb_features=1,
            module__hidden_size=256,
            optimizer__lr=0.0001,
        )
        
    elif(model_name is 'lstm_att'):
        feature_set = 'ohlcv'
        batch_size = 20
        model = NeuralNetRegressor(
            module=SequenceDoubleAtt,
            optimizer=optim.Adam,
            batch_size=batch_size,
            max_epochs=2000, # TODO trovato empiricamente
            train_split=train_split,
            callbacks=cb,

            module__nb_features=5,
            module__hidden_size=256,
            optimizer__lr=0.0001
        )
    elif(model_name is 'lstm_sa'):
        feature_set = 'ext_sa'
        batch_size = 20
        model = NeuralNetRegressor(
            module=SequenceDouble,
            optimizer=optim.Adam,
            batch_size=batch_size,
            max_epochs=2000, # TODO trovato empiricamente
            train_split=train_split,
            callbacks=cb,

            module__nb_features=10,
            module__hidden_size=256,
            optimizer__lr=0.0001
        )
    elif(model_name is 'lstm_sa_att'):
        feature_set = 'ext_sa'
        batch_size = 20
        model = NeuralNetRegressor(
            module=SequenceDoubleAtt,
            optimizer=optim.Adam,
            batch_size=batch_size,
            max_epochs=2000, # TODO trovato empiricamente
            train_split=train_split,
            callbacks=cb,

            module__nb_features=10,
            module__hidden_size=256,
            optimizer__lr=0.0001
        )
    return model, feature_set


    


