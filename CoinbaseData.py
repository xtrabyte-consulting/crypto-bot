import requests
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

class CoinbaseDataCollector:
    PERIODS = {
        60: 'minute',
        300: '5 minute',
        900: '15 minute',
        3600: '1 hour',
        21600: '6 hour',
        86400: '1 day',
    }
    
    def __init__(self, api_key = 'x4qL5kCpgG2jNpue', base_url = 'https://api.exchange.coinbase.com/'):
        self.api_key = api_key
        self.base_url = base_url

    def get_available_trading_pairs(self):
        """
        Fetches all available trading pairs on Coinbase Advanced API.
        """
        url = f'{self.base_url}products'
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f'Failed to fetch data from Coinbase Advanced API: {response.text}')
        data = response.json()
        df = DataFrame(data)
        df.to_csv('trading_pairs.csv')
        return df

    def __get_historical_prices(self, trading_pair, granularity) -> DataFrame:
        """
        Fetches historic rates for a cryptocurrency. Rates are returned in grouped buckets.
        :param trading_pair: Coin pair id (e.g., 'BTC-USD')
        :param granularity: Desired timeslice in seconds
        """
        url = f'{self.base_url}products/{trading_pair}/candles'
        start = datetime.now() - timedelta(seconds=(granularity * 300)) # 300 candles max per request
        end = datetime.now()
        frames = []
        while True:
            print(f'Fetching {self.PERIODS[granularity]} data from {start} to {end}')
            params = {'start': start.isoformat(), 'end': end.isoformat(), 'granularity': granularity}
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f'Cannot fetch data from Coinbase Advanced API before {end}: {response.text}')
                break
            data = response.json()
            if not data:
                break
            df = DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            frames.append(df)
            end = start
            start = start - timedelta(seconds=(granularity * 300))
        prices = pd.concat(frames)
        prices.sort_index(inplace=True)
        prices.to_csv(f'{trading_pair}_{granularity}_prices.csv', index=True)
        return prices
    
    def __get_historical_prices_all_granularities(self, trading_pair) -> [DataFrame]:
        """
        Fetches historic rates for a cryptocurrency at all available timeslices. 
        Rates are returned in grouped buckets.
        :param trading_pair: Coin pair id (e.g., 'BTC-USD')
        """
        granularities = [60, 300, 900, 3600, 21600, 86400]
        frames = []
        for granularity in granularities:
            print(f'Fetching {self.PERIODS[granularity]} data for {trading_pair}')
            df = self.get_historical_prices(trading_pair, granularity)
            frames.append(df)
        return frames
    
    def get_historical_prices(self, trading_pair, granularity):
        """
        Fetches historic rates for a cryptocurrency. Rates are returned in grouped buckets.
        :param trading_pair: Coin pair id (e.g., 'BTC-USD')
        :param granularity: Desired timeslice in seconds
        """
        with ThreadPoolExecutor() as executor:
            return executor.submit(self.__get_historical_prices, trading_pair, granularity)
        
    def get_historical_prices_all_granularities(self, trading_pair):
        """
        Fetches historic rates for a cryptocurrency at all available timeslices. 
        Rates are returned in grouped buckets.
        :param trading_pair: Coin pair id (e.g., 'BTC-USD')
        """
        with ThreadPoolExecutor() as executor:
            return executor.submit(self.__get_historical_prices_all_granularities, trading_pair)
    
    def visualize_historical_prices(self, trading_pair, granularity):
        """
        Visualizes historic rates for a cryptocurrency. Rates are returned in grouped buckets.
        :param trading_pair: Coin pair id (e.g., 'BTC-USD')
        :param granularity: Desired timeslice in seconds
        """
        df = self.get_historical_prices(trading_pair, granularity)
        df.plot(figsize=(20, 10), title=f'{trading_pair} {granularity} seconds')
        return df