"""
Collect prices/features for a symbol in a dataframe fashion
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from master51 import wht
from master51.core.constants import *


SymbolTable = dict[str, "Symbol"]

@dataclass
class Symbol:
    name: str
    full_name: str = field(default="")
    _df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(), repr=False)

    def __post_init__(self):
        if self.full_name == "":
            self.full_name = self.name

    def set_df(self, df: pd.DataFrame):
        assert "open" in df
        assert "high" in df
        assert "low" in df
        assert "close" in df
        self._df = df.copy()
        return self

    def dates_range(self):
        return (self._df.index.values[0], self._df.index.values[-1])

    def add_log_return(self):
        self._df["log_return"] = np.log(self._df["close"]).diff()
        return self

    def add_future(self, key="log_return"):
        self._df["future"] = self._df[key].shift(-1)
        return self

    def add_zprice(self, window_size = 20, include_stats = False, exclude_last = False):
        prices = self._df[OHLC]
        mu = np.zeros_like(prices.close.values, dtype = np.float64)
        sigma = np.ones_like(prices.close.values, dtype = np.float64)
        N = prices.shape[0]
        for i in range(0, N):
            ed = i if exclude_last else i+1  # exclude last if close price is unknown atm
            st = ed - window_size
            if st < 0:
                mu[i] = np.nan
                sigma[i] = np.nan
                continue
            window = prices.iloc[st:ed]
            window = window.values.reshape(-1)
            mu[i] = np.mean(window)
            sigma[i] = np.std(window)
        if include_stats:
            self._df["mu"] = mu
            self._df["sigma"] = sigma
        for tag in OHLC:
            self._df[f"z{tag}"] = (self._df[tag] - mu) / sigma
        return self


@dataclass
class SymbolSet:
    symbols: SymbolTable = field(repr=False)
    name: str = field(default_factory=lambda: "")
    keys: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.keys) == 0:
            for key in self.symbols.keys():
                self.keys.append(key)

    def __iter__(self):
        return iter(self.keys)

    def __getitem__(self, key):
        return self.symbols[key]

    def __len__(self):
        return len(self.keys)

    def build_subset(self, subset_keys, *, name: str | None = None) -> "SymbolSet":
        if name is None:
            name = self.name
        newSymbols = {}
        for key in subset_keys:
            newSymbols[key] = self.symbols[key]
        return SymbolSet(newSymbols, name, subset_keys)

    def collect(self, columns_under_symbols: list[str]):
        # NOTE for time-wise subset use df.loc[t1:t2] is fine
        sub_df_list = []
        for key in self.keys:
            df = self.symbols[key]._df
            sub_df_list.append(df[columns_under_symbols])
        return pd.concat(sub_df_list, axis=1, keys=self.keys)
