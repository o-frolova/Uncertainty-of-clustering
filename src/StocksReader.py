from typing import List, Tuple
import os
import pandas as pd
import numpy as np


class Stocks():
    """A class representing stock data."""
    def __init__(self, id_stock: int, ticker: str, close_prices: list = None, volumes: list = None, dates: list = None) -> None:
        """Initialize a Stocks instance with specified attributes."""
        self.__id_ = id_stock
        self.__ticker = ticker
        self.__close_prices = close_prices if close_prices is not None else []
        self.__volumes = volumes if volumes is not None else []
        self.__dates = dates if dates is not None else []
        self.__returns = self.__get_returns()

    @property
    def id_(self) -> int:
        """Get the stock ID."""
        return self.__id_

    @property
    def ticker(self) -> str:
        """Get the stock ticker."""
        return self.__ticker

    @property
    def close_prices(self) -> list:
        """Get the stock close prices."""
        return self.__close_prices

    @property
    def volumes(self) -> list:
        """Get the stock volumes."""
        return self.__volumes

    @property
    def dates(self) -> list:
        """Get the stock dates."""
        return self.__dates

    @property
    def returns(self) -> pd.Series:
        """Get the stock returns."""
        return self.__returns

    def __get_returns(self) -> pd.Series:
        """Calculating the logarithmic return on an stock """
        if self.__close_prices:
            returns = pd.Series(self.__close_prices).pct_change()
            returns.iloc[0] = 0
            return np.log(1 + returns)
        else:
            return []

    @returns.setter
    def returns(self, returns_):
        """Set stock returns"""
        self.__returns = returns_

    @dates.setter
    def dates(self, dates_):
        """Set stock dates"""
        self.__dates = dates_

    @close_prices.setter
    def close_prices(self, close_prices_):
        """Set stock close_prices"""
        self.__close_prices = close_prices_

    @volumes.setter
    def volumes(self, volumes_):
        """Set stock volumes"""
        self.__volumes = volumes_
    
    @id_.setter
    def id_(self, new_id):
        self.__id_ = new_id


class ReaderStocksData():
    def __init__(self, path:str) -> None:
        """
        Constructor for the ReaderStocksData class.
        Parameters:
        - path (str): The path to the directory containing stock data files.
        """
        self.__path = path

    @property
    def path(self):
        return self.__path

    def __get_stocks_info(self, data) -> Tuple[List, List, List]:
        """
        Extracts stock information (price, volume, date) from the given DataFrame.
        Parameters:
        - data: DataFrame containing stock data.
        Returns:
        - Tuple[List, List, List]: Extracted price, volume, and date lists.
        """
        price, volume, date = [], [], []
        date = list(data['Дата'])
        volume = list(data['Объём'].str.replace(',','.',regex=True))
        # print(data['Цена'])

        try:
            price = list(data['Цена'].str.replace(',','.',regex=True).astype(float))
        except:
            # price = list(data['Цена'].astype(float))
            price = list(data['Цена'].str.replace('.', '').str.replace(',', '.').astype(float))
            # price = list(data['Цена'].str.replace(',', '.').str.replace(',', '.').astype(float))
            
        return price, volume, date

    def __same_length_of_returns(self, Stocks: list) -> list:
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
        for filename in os.listdir(self.__path):

            try:
                f = os.path.join(self.__path, filename)
                if os.path.isfile(f):

                    price, volume, date = [], [], []
                    data = pd.read_csv(f)

                    data['Дата'] = pd.to_datetime(data['Дата'], format="%d.%m.%Y")
                    info_for_exp = data.loc[(date_start <= data['Дата']) & (data['Дата'] <= date_end) ] 

                    price, volume, date = self.__get_stocks_info(info_for_exp)
                    price = price[::-1]
                    volume = volume[::-1]
                    date = date[::-1]

                    DATA_OF_STOCKS.append(Stocks(count, filename[:-4], price, volume, date))

                    count += 1
            except:
                continue

        for stock in DATA_OF_STOCKS:
                TICKERS.append(stock.ticker)

        DATA_OF_STOCKS = self.__same_length_of_returns(DATA_OF_STOCKS)

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
    
