from tkinter import N
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
from keras.optimizers import Adam, SGD
from kerastuner.tuners import Tuner, RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from pickle import dump,load
import warnings
warnings.filterwarnings("ignore")

class LSTMModelTuning:
    def __init__(self, prices: DataFrame, split_ratio = 0.8, inputs = 120, outputs = 24):
        self.prices = prices
        self.series = prices['close'].values.reshape(-1,1)
        self.train, self.test = self.split(self.series, split_ratio)
        self.scaler, self.scaled_train, self.scaled_test = self.scale(self.train, self.test)
        self.X_train, self.Y_train = self.create_supervised(self.scaled_train, inputs, outputs)
        self.X_test, self.Y_test = self.create_supervised(self.scaled_test, inputs, outputs)        
    
    def split(self, series: np.ndarray, split_ratio = 0.8) -> (DataFrame, DataFrame):
        self.split_index= math.floor(len(self.series) * split_ratio)
        self.train, self.test = series[:self.split_index,:], series[self.split_index:,:]
        return self.train, self.test
    
    def create_supervised(self, series: DataFrame, inputs = 120, outputs = 24) -> (np.array, np.array):
        X, Y = [], []
        for i in range(len(series) - (inputs + outputs + 1)):
            X.append(series[i:(i+inputs), 0])
            Y.append(series[(i+inputs):(i+inputs+outputs), 0])
        X = np.array(X)
        Y = np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))
        return X, Y
    
    def scale(self, train: np.array, test: np.array, scaler = MinMaxScaler(feature_range = (0, 1))):
        self.scaler = scaler.fit(train)
        self.scaled_train = scaler.transform(train)
        self.scaled_test = scaler.transform(test)
        return self.scaler, self.scaled_train, self.scaled_test
    
    def build_model(self, hp: HyperParameters):
        print(f'Input Shape: {self.X_train.shape[1], self.X_train.shape[2]}')
        print(f'Output Shape: {self.Y_train.shape[1], self.Y_train.shape[2]}')
        model = Sequential()
        model.add(LSTM(units = hp.Int('input_units', min_value = 50, max_value = 500, step = 50), 
                       return_sequences = True, 
                       input_shape = (120, 1)))
        model.add(Dropout(rate = hp.Float('dropout', min_value = 0.1, max_value = 0.5, step = 0.1)))
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(LSTM(units = hp.Int(f'units_{i}', min_value = 50, max_value = 500, step = 50), 
                           return_sequences = True))
            model.add(Dropout(rate = hp.Float(f'dropout_{i}', min_value = 0.1, max_value = 0.5, step = 0.1)))
        model.add(LSTM(units = hp.Int(f'units_last', min_value = 50, max_value = 500, step = 50), 
                           return_sequences = False))
        model.add(Dense(24, activation = hp.Choice('activation', values=['relu', 'sigmoid'], default='relu')))
        model.compile(optimizer = Adam(learning_rate=0.1), loss = 'mean_squared_error', metrics = ['mse'])
        return model
    
    def tune_model(self, epochs = 10, max_trials = 10, executions_per_trial = 3, directory = 'tuning'):
        tuner = RandomSearch(self.build_model, objective = 'mse', max_trials = max_trials, executions_per_trial = executions_per_trial, directory = directory)
        tuner.search_space_summary()
        tuner.search(self.X_train, self.Y_train, epochs = epochs, validation_data = (self.X_test, self.Y_test))
        tuner.results_summary()
        self.best_model = tuner.get_best_models(num_models = 1)[0]
        self.best_hyperparameters = tuner.get_best_hyperparameters(num_trials = 1)[0]
        return self.best_model, self.best_hyperparameters
    
    def save(self, model, filename):
        save_model(model, f'{filename}.h5')
        model.save_weights(f'{filename}_weights.h5')
        with open(f'{filename}.json', 'w') as f:
            f.write(model.to_json())
        dump(self.scaler, open(f'{filename}_scaler.pkl', 'wb'))
        
    def load_model(self, filename):
        with open(f'{filename}.json', 'r') as f:
            model = model_from_json(f.read())
        if model is None:
            raise Exception('Model not found')
        model.load_weights(f'{filename}_weights.h5')
        self.scaler = load(open(f'{filename}_scaler.pkl', 'rb'))
        return model
    
    def predict(self, X = None):
        if X is None:
            X = self.X_test
        return self.scaler.inverse_transform(self.best_model.predict(X))
        