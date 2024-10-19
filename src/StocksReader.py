from typing import List, Tuple
import os
import pandas as pd
import pathlib
from src.Stocks import Stocks

class ReaderStocksData():
    def __init__(self, path: pathlib.Path) -> None:
        """
        Constructor for the ReaderStocksData class.
        Parameters:
        - path (str): The path to the directory containing stock data files.
        """
        self._path = path

    @property
    def path(self):
        return self._path

    def _get_stocks_info(self, data) -> Tuple[List, List, List]:
        """
        Extracts stock information (price, volume, date) from the given DataFrame.
        Parameters:
        - data: DataFrame containing stock data.
        Returns:
        - Tuple[List, List, List]: Extracted price, volume, and date lists.
        """
        price, volume, date = [], [], []
        date = list(data['Date'])
        volume = list(data['Volume'])
        price = list(data['Close'])
        return price, volume, date

    def _same_length_of_returns(self, Stocks: list) -> list:
        """
        Brings information on the returns of each asset to the same length
        Parameters:
        - Stocks (list): List of stock objects.
        Returns:
        - list: Updated list of stock objects with returns adjusted to the minimum length.
        """

        mimimum = float('inf')
        for stock in Stocks:
            if len(stock.returns) < mimimum:
                mimimum = len(stock.returns)
        for stock in Stocks:
            stock.returns = stock.returns[:mimimum - 1]
            stock.dates = stock.dates[:mimimum - 1]
            stock.close_prices = stock.close_prices[:mimimum - 1]
            stock.volumes = stock.volumes[:mimimum - 1]

        return Stocks

    def load_data(self, date_start: str, date_end:str) -> Tuple[List, List, List]:
        """
        Loads stock data from files within the specified date range.
        Parameters:
        - date_start (str): Start date for loading stock data.
        - date_end (str): End date for loading stock data.
        Returns:
        - Tuple[List, List, List]: Data for building the model, data for evaluation, and list of tickers.
        """
        TICKERS = []
        DATA_OF_STOCKS = []

        count = 0
        for filename in os.listdir(self._path):

            try:
                f = os.path.join(self._path, filename)
                if os.path.isfile(f):

                    price, volume, date = [], [], []
                    data = pd.read_csv(f)

                    data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
                    info_for_exp = data.loc[(date_start <= data['Date']) & (data['Date'] <= date_end) ] 

                    price, volume, date = self._get_stocks_info(info_for_exp)
                    price = price[::-1]
                    volume = volume[::-1]
                    date = date[::-1]

                    DATA_OF_STOCKS.append(Stocks(count, filename[:-4], price, volume, date))

                    count += 1
            except:
                continue

        for stock in DATA_OF_STOCKS:
                TICKERS.append(stock.ticker)

        DATA_OF_STOCKS = self._same_length_of_returns(DATA_OF_STOCKS)

        return DATA_OF_STOCKS, TICKERS


# class ReaderStocksDatayfinance():
#     def __init__(self, path:str) -> None:
#         """
#         Constructor for the ReaderStocksData class.
#         Parameters:
#         - path (str): The path to the directory containing stock data files.
#         """
#         self.__path = path

#     @property
#     def path(self):
#         return self.__path

#     def __same_length_of_returns(self, Stocks: list) -> list:
#         """
#         Brings information on the returns of each asset to the same length
#         Parameters:
#         - Stocks (list): List of stock objects.
#         Returns:
#         - list: Updated list of stock objects with returns adjusted to the minimum length.
#         """
#         mimimum = float('inf')
#         for stock in Stocks:
#             if len(stock.returns) < mimimum:
#                 mimimum = len(stock.returns)
#         for stock in Stocks:
#             stock.returns = stock.returns[:mimimum - 1]
#             stock.dates = stock.dates[:mimimum - 1]
#             stock.close_prices = stock.close_prices[:mimimum - 1]
#             stock.volumes = stock.volumes[:mimimum - 1]

#         return Stocks

#     def load_data(self, date_start: str, date_end:str) -> Tuple[List, List, List]:
#         """
#         Loads stock data from files within the specified date range.
#         Parameters:
#         - date_start (str): Start date for loading stock data.
#         - date_end (str): End date for loading stock data.
#         Returns:
#         - Tuple[List, List, List]: Data for building the model, data for evaluation, and list of tickers.
#         """
#         TICKERS = []
#         DATA_OF_STOCKS = []
#         count = 0
#         for filename in os.listdir(self.__path):
#             try:
#                 f = os.path.join(self.__path, filename)
#                 if os.path.isfile(f):
#                     data = pd.read_csv(f)
#                     DATA_OF_STOCKS.append(Stocks(count, filename[:-4], list(data['Close']), list(data['Volume']),  data['Date']))
#                     count += 1
#             except:
#                 continue

#         for stock in DATA_OF_STOCKS:
#                 TICKERS.append(stock.ticker)


#         DATA_OF_STOCKS = self.__same_length_of_returns(DATA_OF_STOCKS)

#         return DATA_OF_STOCKS.copy(), TICKERS.copy()
    
