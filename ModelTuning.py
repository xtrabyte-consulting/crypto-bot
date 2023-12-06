import pandas as pd
import math
import keras
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import save_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from pickle import dump,load
import warnings
warnings.filterwarnings("ignore")

class LSTMModelTuning:
    def __init__(self, prices: DataFrame):
        self.series = prices['close'].values.reshape(-1,1)
        
    def scale(self, scaler = MinMaxScaler(feature_range = (0, 1))):
        self.scaled_data = scaler.fit_transform(self.series)
        return self.scaled_data
    
    def split(self, train_size = 0.8):
        self.train_size = int(len(self.scaled_data) * train_size)
        self.test_size = len(self.scaled_data) - self.train_size
        self.train, self.test = self.scaled_data[0:self.train_size,:], self.scaled_data[self.train_size:len(self.scaled_data),:]
        self.train = self
        return self.train, self.test
        