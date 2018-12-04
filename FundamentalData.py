import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from datetime import timedelta


class FundamentalData:

    def __init__(self):
        self.fundamentals = pd.read_csv('nyse/fundamentals.csv', index_col=0, header=0).sort_index()
        self.prices = pd.read_csv('nyse/prices-split-adjusted.csv', index_col=[1,0], header=0).sort_index()

        format_str = '%Y-%m-%d'
        period_end_prices = []
        for i in self.fundamentals.index:
            symbol = self.fundamentals.at[i, "Ticker Symbol"]
            period_ending = self.fundamentals.at[i, "Period Ending"]

            if self.prices.index.contains(symbol):
                prices_by_date = self.prices.xs(symbol, level="symbol")

                found = False
                limit_exceeded = False
                next_date = datetime.strptime(period_ending, format_str) + timedelta(days=30) # Add 1 month delay
                next_date_str = next_date.strftime(format_str)

                while not found and not limit_exceeded:
                    if prices_by_date.index.contains(next_date_str):
                        found = True
                    else:
                        next_date = datetime.strptime(next_date_str, format_str) + timedelta(days=1)
                        if next_date > datetime(2019,1,1):
                            limit_exceeded = True
                        next_date_str = next_date.strftime(format_str)

                if not limit_exceeded:
                    price = prices_by_date.at[next_date_str, "close"]
                else:
                    price = 0
            else:
                price = 0
            period_end_prices.append(price)

        self.fundamentals["Period End Prices"] = pd.Series(period_end_prices, index=self.fundamentals.index)

        self.fundamentals["P/E Ratio"] = self.fundamentals["Period End Prices"] / self.fundamentals["Earnings Per Share"]

        self.buildExamples()

    def buildExamples(self):
        self.names = []
        self.X = []
        self.Y = []
        symbols = np.unique(self.fundamentals["Ticker Symbol"])
        # filter symbols with complete dataFrame
        for name, group in self.fundamentals.groupby(by=["Ticker Symbol"]):
            if len(group) == 4:
                self.names.append(name)
                self.X.append(
                    np.concatenate([
                        np.array(group["Quick Ratio"]),
                        np.array(group["P/E Ratio"]),
                        np.array(group["Profit Margin"]),
                        np.array(group["Pre-Tax ROE"]),
                    ])
                )
                prices = np.array(group["Period End Prices"])
                self.Y.append(
                        np.array([prices[3] > prices[0], prices[3] <= prices[0]]),
                )
        self.X = np.nan_to_num(np.reshape(self.X, (440,16)))
        self.Y = np.reshape(self.Y, (440,2))
