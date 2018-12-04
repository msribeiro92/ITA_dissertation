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

        period_end_prices = []
        next_month_prices = []
        for i in self.fundamentals.index:
            period_end_prices.append(self.findDate(i))
            next_month_prices.append(self.findDate(i, delay=30))

        self.fundamentals["Period End Prices"] = pd.Series(period_end_prices, index=self.fundamentals.index)
        self.fundamentals["Next Month Prices"] = pd.Series(next_month_prices, index=self.fundamentals.index)

        self.fundamentals["P/E Ratio"] = self.fundamentals["Period End Prices"] / self.fundamentals["Earnings Per Share"]

        self.buildExamples()

    def findDate(self, i, delay=0):
        format_str = '%Y-%m-%d'
        symbol = self.fundamentals.at[i, "Ticker Symbol"]
        period_ending = self.fundamentals.at[i, "Period Ending"]

        if self.prices.index.contains(symbol):
            prices_by_date = self.prices.xs(symbol, level="symbol")
            found = False
            limit_exceeded = False
            next_date = datetime.strptime(period_ending, format_str) + timedelta(days=delay) # Add delay
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
                return prices_by_date.at[next_date_str, "close"]
            else:
                return 0
        else:
            return 0

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
                period_end_prices = np.array(group["Period End Prices"])
                next_month_prices = np.array(group["Next Month Prices"])
                self.Y.append(
                    np.array([
                        next_month_prices[3] > period_end_prices[3],
                        next_month_prices[3] <= period_end_prices[3]
                    ]),
                )
        self.X = np.nan_to_num(np.reshape(self.X, (440,16)))
        self.Y = np.reshape(self.Y, (440,2))
