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
from sa_tools import *



def save(model, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(model, f)
        
def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    


def gain(C, C_pred, opn):
    O = opn.reindex_like(C)
    CO_diff = C - O
    growth = C_pred > O
    decline = C_pred < O
    return CO_diff[growth].sum() - CO_diff[decline].sum()


def roi(C, C_pred, opn):
    mean_opn = opn.reindex_like(C).mean()
    return gain(C, C_pred, opn) / mean_opn

#aggiunge le date mancanti usando come valore quello medio della precedente e della successiva note
def fill_dates(df):
    dates = pd.date_range(start=df.index.min(), end=df.index.max())
    for date in dates:
        if(date not in df.index):
            df.loc[date] = None
    df = df.sort_index()
    return (df.fillna(method='ffill') + df.fillna(method='bfill'))/2

    
#returns open high low close volume
def get_index(index="^DJI", start_date="2000-01-01", end_date="2018-12-31"):
    security = yfinance.Ticker(index)# TODO trova mercato asiatico e indiano
    security_data = security.history(start=start_date, end=end_date, actions=False)
    return security_data


def split_index(security_data):
    return security_data["Open"], security_data["High"], security_data["Low"], security_data["Close"], security_data["Volume"]


def prepare_data(features, target):
    X = pd.DataFrame(features)
    X.dropna(inplace=True)
    Y = target.reindex_like(X)
    return X, Y

def days_group(data, n_days=10):
    res = np.zeros([data.shape[0]-n_days, n_days, data.shape[1]], dtype=np.float32)
    for i, el in enumerate(data):
        if(i >= n_days):
            res[i-n_days] = data[i-n_days:i]
    return res


def get_ext_feats(ohlcv):
    res = ohlcv.copy()
    
    opn = res["Open"]
    close = res["Close"]
    high = res["High"]
    low = res["Low"]
    volume = res["Volume"]
    
    #calucate derived indicators
    TP = ((high + low + close) / 3 )
    trs = pd.DataFrame(index=high.index)
    trs['tr0'] = abs(high - close)
    trs['tr1'] = abs(high - close)
    trs['tr2'] = abs(low - close)
    TR = trs[['tr0', 'tr1', 'tr2']].max(axis=1)
    ema20 = opn.ewm(span=20).mean()
    ma10 = opn.rolling(window=10).mean()
    ma5 = opn.rolling(window=5).mean()
    macd = opn.ewm(span=26).mean() - opn.ewm(span=12).mean()
    cci_ndays=20
    cci = (TP - TP.rolling(cci_ndays).mean()) / (0.015 * TP.rolling(cci_ndays).std())
    atr = TR.ewm(span = 10).mean()
    ma20 = opn.rolling(window=20).mean()
    std20 = opn.rolling(window=20).std()
    k=2
    boll_up =  ma20 + (k*std20)
    boll_down = ma20 - (k*std20)
    roc = (opn - opn.shift(9))/opn.shift(9)
    mtm6 = (opn - opn.shift(127))
    mtm12 = (opn - opn.shift(253)) #length of a trading year is on average 253 days
    wvad = (((close - low) - (high - close)) * volume/(high - low))
    smi = (close - (high - low)/2)
    
    res["EMA20"] = ema20
    res["MA5"] = ma5
    res["MA10"] = ma10
    res["MA20"] = ma20
    res["MACD"] = macd
    res["CCI"] = cci
    res["ATR"] = atr
    res["BollUp"] = boll_up
    res["BollDown"] = boll_down
    res["WVAD"] = wvad
    res["MTM6"] = mtm6
    res["MTM12"] = mtm12
    res["SMI"] = smi
    res["ROC"] = roc
    
    return res


# takes as input ohlcv dataframe
def get_data_set(ohlcv, ext_feats=True, usd_index='DX-Y.NYB', wavelet=True):
    feats = ohlcv.copy()
    
    feats["Close"] = feats["Close"].shift(1)
    feats["High"] = feats["High"].shift(1)
    feats["Low"] = feats["Low"].shift(1)
    feats["Volume"] = feats["Volume"].shift(1)
    feats = feats.dropna()
    
    usd_open = fill_dates(get_index(usd_index, start_date=feats["Open"].index.min(), end_date=feats["Open"].index.max())["Open"])
    
    if(wavelet):
    #apply transforms
        for f_name in ('Open', 'Close', 'High', 'Low'):
            feats[f_name] = apply_wavelet_transform(feats[f_name]) 
        usd_open = apply_wavelet_transform(usd_open)
        feats = feats.dropna()
     
    
    
    if(ext_feats):
        feats = get_ext_feats(feats)
        feats["USDOpen"]  = usd_open
    feats = feats.dropna()
    
    return feats


def get_dataset_by_name(ohlcv, name):
    if(name is "open"):
        return get_data_set(ohlcv, ext_feats=False)[["Open"]]
    elif(name is "ohlcv"):
        return get_data_set(ohlcv, ext_feats=False)
    elif(name is "ext"):
        return get_data_set(ohlcv)
    raise Exception('Nome del feature-set non valido')
    
    
def apply_wavelet_transform(data, consider_future=False, wavelet='haar'):
    res = data.copy()
    if(consider_future):
        res, _ = pywt.dwt(data.copy(), wavelet=wavelet)
    else:
        for i in range(res.shape[0]):
            if(i > 0):
                cA =  waveletSmooth(data.iloc[:i+1].copy(), wavelet=wavelet, level=4, DecLvl=3)
                res.iloc[i] = cA[-1]
    return res


def get_dataset_train(datasets, feature_set, train_dates, val_dates, index='^GSPC', sa=None, ld=False):   
    
    y = datasets[index]["target"]
    dataset = datasets['^GSPC']["features"][feature_set]
       
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_train, x_val = x_scaler.fit_transform(dataset.iloc[train_dates].to_numpy(np.float32)), x_scaler.transform(dataset.iloc[val_dates].to_numpy(np.float32))
    y_train, y_val = y_scaler.fit_transform(y.iloc[train_dates].to_numpy(np.float32)[...,None]), y_scaler.transform(y.iloc[val_dates].to_numpy(np.float32)[...,None])
    
    if(sa is not None):
        if(not ld):
            encoder = get_encoder(x_train, x_val, sa_hidden_size=10)
            save(encoder, sa)
        encoder = load(sa)
        x_train, x_val = encode(x_train, encoder), encode(x_val, encoder)
        
    x = np.concatenate((x_train, x_val))
    y = np.concatenate((y_train, y_val))
    
    return x, y, x_train, x_val, x_scaler, y_train, y_val, y_scaler
    


def get_datasets(indices = ['^GSPC'],  feature_sets = ['open', 'ohlcv', 'ext'], start_date="2000-01-01", end_date="2018-12-31"):
    start_data = None
    end_data = None
    datasets = {}
    for index in indices:
        datasets[index] = {}
        datasets[index]["original"] = get_index(index=index, start_date=start_date, end_date=end_date)
        datasets[index]["original"] = fill_dates(datasets[index]["original"].copy())
        datasets[index]["features"] = {}
        datasets[index]["target"] = datasets[index]["original"]["Close"].copy()
#         print(datasets)
        for feature_set in feature_sets:
            data = get_dataset_by_name(datasets[index]["original"].copy(), name=feature_set)
            data = data.dropna()
            if(start_data is None):
                start_data = data.index.min()
            else:
                start_data = max(data.index.min(), start_data)
            if(end_data is None):
                end_data = data.index.max()
            else:
                end_data = min(data.index.max(), end_data)
            datasets[index]["features"][feature_set] =  data.copy()

    #allineamento delle date
    for index in datasets.keys():      
        datasets[index]["target"] = datasets[index]["target"].loc[start_data:end_data].copy()
        for feature_set in datasets[index]["features"].keys():  
            datasets[index]["features"][feature_set] = datasets[index]["features"][feature_set].loc[start_data:end_data].copy()
    
    return datasets
