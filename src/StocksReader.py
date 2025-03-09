import logging
import pathlib
from typing import List, Tuple

import pandas as pd

from src.Stocks import Stocks


class ReaderStocksData:
    """
    A class to read and process stock data from CSV files.

    Attributes:
        path (pathlib.Path): The path to the directory containing stock data files.

    Methods:
        load_data(date_start: str, date_end: str) -> Tuple[List[Stocks], List[str]]:
            Loads stock data from CSV files within the specified date range.

        _get_stocks_info(data) -> Tuple[List[float], List[float], List[str]]:
            Extracts stock price, volume, and date information from the DataFrame.

        _same_length_of_returns(Stocks: List[Stocks]) -> List[Stocks]:
            Trims the returns, dates, close prices, and volumes of the stocks
            to ensure they are of the same length.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """
        Initializes the ReaderStocksData with the specified directory path.

        Args:
            path (pathlib.Path): The path to the directory containing stock data files.
        """
        self._path = path

    @property
    def path(self):
        """
        Returns the path to the stock data directory.

        Returns:
            pathlib.Path: The path to the directory containing stock data files.
        """
        return self._path

    def _get_stocks_info(self, data) -> Tuple[List[float], List[float], List[str]]:
        """
        Extracts stock information from the given DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing stock data.

        Returns:
            Tuple[List[float], List[float], List[str]]: A tuple containing:
                - List of stock closing prices as floats.
                - List of stock volumes as floats.
                - List of stock dates as strings.
        """
        price = data["Close"].astype(float).tolist()
        volume = data["Volume"]
        date = data["Date"].tolist()
        return price, volume, date

    def _same_length_of_returns(self, Stocks: List[Stocks]) -> List[Stocks]:
        """
        Trims the returns and associated data of stocks to the same length.

        Args:
            Stocks (List[Stocks]): A list of Stock objects.

        Returns:
            List[Stocks]: A list of Stock objects with trimmed returns and data.
        """
        minimum = min(len(stock.returns) for stock in Stocks)
        for stock in Stocks:
            stock.returns = stock.returns[: minimum - 1]
            stock.dates = stock.dates[: minimum - 1]
            stock.close_prices = stock.close_prices[: minimum - 1]
            stock.volumes = stock.volumes[: minimum - 1]
        return Stocks

    def load_data(
        self, date_start: str, date_end: str
    ) -> Tuple[List[Stocks], List[str]]:
        """
        Loads stock data from CSV files within the specified date range.

        Args:
            date_start (str): The start date in 'YYYY-MM-DD' format.
            date_end (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            Tuple[List[Stocks], List[str]]: A tuple containing:
                - List of Stock objects for the specified date range.
                - List of tickers corresponding to the loaded stock data.
        """
        TICKERS = []
        DATA_OF_STOCKS = []

        count = 0
        for filename in pathlib.Path(self._path).iterdir():
            if filename.is_file():
                try:
                    data = pd.read_csv(filename)
                    data["Date"] = pd.to_datetime(
                        data["Date"], format="%Y-%m-%d", errors="coerce"
                    )
                    data.dropna(subset=["Date"], inplace=True)

                    # Filter directly using datetime objects
                    filtered_data = data[
                        (data["Date"] >= date_start) & (data["Date"] <= date_end)
                    ]

                    if filtered_data.empty:
                        continue

                    price, volume, date = self._get_stocks_info(filtered_data)
                    DATA_OF_STOCKS.append(
                        Stocks(count, filename.stem, price, volume, date)
                    )
                    count += 1

                except Exception as e:
                    logging.error(f"Error processing file {filename}: {e}")
                    continue

        for stock in DATA_OF_STOCKS:
            TICKERS.append(stock.ticker)

        DATA_OF_STOCKS = self._same_length_of_returns(DATA_OF_STOCKS)
        return DATA_OF_STOCKS, TICKERS
