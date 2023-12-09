import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler


class PeriodAnalysis:
    def __init__(self, time_series):
        self.time_series = np.array(time_series)
    
    def apply_fourier_transform(self):
        self.transformed = fft(self.time_series)
        self.frequencies = fftfreq(len(self.time_series))
        self.amplitudes = np.abs(self.transformed)
        self.phases = np.angle(self.transformed)
        self.df = pd.DataFrame({'frequency': self.frequencies, 'amplitude': self.amplitudes, 'phase': self.phases})
        return self.transformed
    
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
        return self.df[self.df['amplitude'] > self.amplitudes.mean() + (std_devs * self.amplitudes.std())]
    
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
