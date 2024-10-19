import pandas as pd
import numpy as np

class Stocks():
    """
    A class representing stock data.
    """
    def __init__(
            self,
            id_stock: int,
            ticker: str,
            close_prices: list = None,
            volumes: list = None,
            dates: list = None
    ) -> None:
        """
        Initialize a Stocks instance with specified attributes.
        """
        self._id_ = id_stock
        self._ticker = ticker
        self._close_prices = close_prices if close_prices is not None else []
        self._volumes = volumes if volumes is not None else []
        self._dates = dates if dates is not None else []
        self._returns = self.__get_returns()
        self._weight_cluster = {'single': None,'complete': None, 'average': None, 'ward': None, 'DBHT': None}
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

    def __get_returns(self) -> pd.Series:
        """
        Calculating the non-logarithmic return on an stock.
        """
        if self._close_prices:
            returns = pd.Series(self._close_prices).pct_change()
            returns.iloc[0] = 0
            return np.log(1 + returns)
        else:
            return []

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