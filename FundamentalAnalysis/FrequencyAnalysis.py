import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.fft import fft
from scipy.fft import fft
from scipy.signal import fold


class TimeSeriesAnalysis:
    def __init__(self, time_series):
        self.time_series = np.array(time_series)
    
    def scipy_fourier_transform(self):
        return fft(self.time_series)
    
    def epoch_folding(self, period):
        return fold(self.time_series, period)
