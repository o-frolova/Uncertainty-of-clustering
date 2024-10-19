from typing import List

import numpy as np
import pandas as pd


class Stocks():
    """
    A class representing stock data for financial analysis.

    This class encapsulates the essential attributes of a stock, including its ID, ticker symbol,
    closing prices, volumes, dates, and calculated returns. It provides methods for accessing and 
    manipulating this data, enabling efficient financial analysis and modeling.

    Attributes:
        id_ (int): The unique identifier for the stock.
        ticker (str): The ticker symbol of the stock.
        close_prices (list): A list of closing prices for the stock over time.
        volumes (list): A list of trading volumes for the stock over time.
        dates (list): A list of dates corresponding to the closing prices and volumes.
        returns (pd.Series): A pandas Series containing the calculated returns based on closing prices.
        weight_cluster (pd.Series): A pandas Series representing the weight of the stock in a specific cluster.

    Methods:
        _get_returns() -> pd.Series:
            Calculates the non-logarithmic returns based on closing prices.

        id_:
            Gets or sets the stock ID.
        
        ticker:
            Gets the stock ticker.
        
        close_prices:
            Gets or sets the list of closing prices.
        
        volumes:
            Gets or sets the list of trading volumes.
        
        dates:
            Gets or sets the list of dates.
        
        returns:
            Gets or sets the calculated stock returns.
        
        weight_cluster:
            Gets or sets the stock weight in a cluster.
    """
    def __init__(self, id_stock: int, ticker: str, close_prices: List[float] = None, 
                 volumes: List[float] = None, dates: List[str] = None) -> None:
        self._id_ = id_stock
        self._ticker = ticker
        self._close_prices = close_prices if close_prices is not None else []
        self._volumes = volumes if volumes is not None else []
        self._dates = dates if dates is not None else []
        self._returns = self._get_returns()
        self._returns_temp = []

    @property
    def id_(self) -> int:
        """
        Get the stock ID.
        """
        return self._id_

    @property
    def ticker(self) -> str:
        """
        Get the stock ticker.
        """
        return self._ticker

    @property
    def close_prices(self) -> list:
        """
        Get the stock close prices.
        """
        return self._close_prices

    @property
    def volumes(self) -> list:
        """
        Get the stock volumes.
        """
        return self._volumes

    @property
    def dates(self) -> list:
        """
        Get the stock dates.
        """
        return self._dates

    @property
    def returns(self) -> pd.Series:
        """
        Get the stock returns.
        """
        return self._returns

    @property
    def weight_cluster(self) -> pd.Series:
        """
        Get the stock weight_cluster.
        """
        return self._weight_cluster

    def _get_returns(self) -> pd.Series:
        """
        Calculate non-logarithmic returns for the stock.
        """
        if len(self._close_prices) > 1:
            returns = pd.Series(self._close_prices).pct_change()
            returns.iloc[0] = 0
            return np.log(1 + returns)
        return pd.Series([])

    @returns.setter
    def returns(self, returns_):
        """
        Set stock returns
        """
        self._returns = returns_

    @dates.setter
    def dates(self, dates_):
        """
        Set stock dates
        """
        self._dates = dates_

    @close_prices.setter
    def close_prices(self, close_prices_):
        """
        Set stock close_prices
        """
        self._close_prices = close_prices_

    @volumes.setter
    def volumes(self, volumes_):
        """
        Set stock volumes
        """
        self._volumes = volumes_

    @weight_cluster.setter
    def weight_cluster(self, weight_cluster_):
        """
        Set stock weight_cluster
        """
        self._weight_cluster = weight_cluster_
    
    @id_.setter
    def id_(self, new_id):
        """
        Set id stock
        """
        self._id_ = new_id