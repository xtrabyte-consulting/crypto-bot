from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class PeriodAnalysis:
    def __init__(self, time_series):
        self.time_series = np.array(time_series)
        
    def scale_series(self):
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.time_series = self.scaler.fit_transform(self.time_series.reshape(-1,1)).flatten()
        return self.time_series
    
    def apply_fourier_transform(self):
        self.transformed = fft(self.time_series)
        self.frequencies = fftfreq(len(self.time_series))
        self.amplitudes = np.abs(self.transformed)
        self.phases = np.angle(self.transformed)
        self.df = pd.DataFrame({'frequency': self.frequencies, 'amplitude': self.amplitudes, 'phase': self.phases})
        return self.transformed
    
    def scale_and_transform(self):
        self.scale_series()
        self.apply_fourier_transform()
        return self.transformed
    
    def disalplay_time_series_statistics(self):
        print(f'Number of observations: {len(self.time_series)}')
        print(f'Price mean: {self.time_series.mean()}')
        print(f'Price standard deviation: {self.time_series.std()}')
        print(f'Price max: {self.time_series.max()}')
        print(f'Price min: {self.time_series.min()}')
        print(f'Price median: {np.median(self.time_series)}')
    
    def display_transform_statistics(self):
        print(f'Number of observations: {len(self.time_series)}')
        print(f'Frequency range: {np.min(self.frequencies)} - {np.max(self.frequencies)}')
        print(f'Amplitude mean: {self.amplitudes.mean()}')
        print(f'Amplitude standard deviation: {self.amplitudes.std()}')
        print(f'Amplitude max: {self.amplitudes.max()}')
        print(f'Amplitude min: {self.amplitudes.min()}')
        print(f'Amplitude median: {np.median(self.amplitudes)}')
        
    def plot_transformed(self, start = 0, end = None):
        sns.scatterplot(x='frequency', y='amplitude', data=self.df.iloc[start:end])
        plt.show()
    
    def sort_by_ampltidue(self):
        df = self.df[self.df['amplitude'] > 0]
        df = df.sort_values('amplitude', ascending=False)
        return df
    
    def plot_largest_amplitudes(self, top_n = 10):
        df = self.sort_by_ampltidue()
        df = df.head(top_n)
        sns.barplot(x='frequency', y='amplitude', data=df)
        plt.show()
        
    def extract_statistically_significant_periods(self, std_devs = 2):
        self.df_significant = self.df[self.df['amplitude'] > self.amplitudes.mean() + (std_devs * self.amplitudes.std())]
        return self.df_significant
    
    def plot_statistically_significant_periods(self):
        self.extract_statistically_significant_periods()
        sns.scatterplot(x='frequency', y='amplitude', data=self.df_significant)
        plt.show()
        
    def fold_on_significant_periods(self):
        self.df_significant = self.extract_statistically_significant_periods()
        self.df_significant = self.df_significant[self.df_significant['frequency'] > 0]
        self.df_significant = self.df_significant.sort_values('frequency')
        self.df_significant['period'] = 1 / self.df_significant['frequency']
        self.df_significant['period'] = self.df_significant['period'].astype(int)
        self.df_significant = self.df_significant[self.df_significant['period'] > 1]
        self.df_significant = self.df_significant.drop_duplicates(subset='period')
        self.folds = []
        for period in self.df_significant['period'].values:
            self.folds.append(self.epoch_fold(period))
        return self.df_significant
    
    def plot_folded(self):
        self.fold_on_significant_periods()
        for fold in self.folds:
            plt.plot(fold)
            plt.show()
    
    def decompose(self, period):
        self.decomposition = seasonal_decompose(self.time_series.reshape(-1, 1), period=period)
        return self.decomposition
    
    def plot_decomposition(self, period):
        self.decomposition = self.decompose(period)
        plt.plot(self.time_series, label='Price')
        plt.plot(self.decomposition.trend, label='Trend')
        plt.plot(self.decomposition.seasonal, label='Seasonality')
        plt.plot(self.decomposition.resid, label='Residual')
        plt.legend()
        plt.show()
        
    def fit_lin_reg(self):
        self.linear_regression = LinearRegression()
        x = np.arange(len(self.time_series)).reshape(-1, 1)
        self.linear_regression.fit(x, self.time_series.reshape(-1, 1))
        self.lin_reg_trend = self.linear_regression.predict(x)
        return self.lin_reg_trend
    
    def plot_lin_reg(self):
        self.fit_lin_reg()
        plt.plot(self.time_series, label='Price')
        plt.plot(self.lin_reg_trend, label='Trend')
        plt.legend()
        plt.show()
        
    def detrend_linear(self):
        self.fit_lin_reg()
        self.detrended = self.time_series - self.lin_reg_trend.flatten()
        return self.detrended
    
    def plot_detrended(self):
        self.detrend_linear()
        plt.plot(self.detrended)
        plt.show()
    
    def epoch_fold(self, period):
        """
        Fold the time series into epochs of the given period.
        Args:
            period (int): Length of the period to fold the time series over.

        Returns:
            Series: The folded time series.
        """
        #self.epoch_folded = pd.Series(self.time_series).groupby(pd.Series(self.time_series).index // period).mean()
        self.epoch_folded = np.zeros(period)
        for i in range(len(self.time_series)):
            self.epoch_folded[i % period] += self.time_series[i]
        return self.epoch_folded
