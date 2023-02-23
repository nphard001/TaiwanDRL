import numpy as np
import pandas as pd
from master51 import wht
from master51.core.dataset.symbol import Symbol, SymbolTable

translation = {
    '收盤價(元)': 'close',
    '最低價(元)': 'low',
    '最高價(元)': 'high',
    '開盤價(元)': 'open',
}

class TEJParser:
    """ csv -> Symbols & OHLC Database """
    def __init__(self, csv_file: str):
        # read csv and build date index
        raw_df = pd.read_csv(csv_file, encoding='big5', header=[0, 1])
        dates = []
        for v in raw_df.iloc[:, 0]:
            dt = wht.tw_datetime(v, format="%m/%d/%Y")
            dt = dt.replace(hour=9)
            dates.append(dt)
        dates = pd.Index(dates, name='dates')
        raw_df = raw_df.set_index(dates)
        del raw_df[raw_df.columns[0]]
        self._df = raw_df
        self.symbols: SymbolTable = {}

        # collect symbol set
        for full_name in self._df.columns.levels[0]:
            collected = {}
            for sub_name in self._df.columns.levels[1]:
                if (full_name, sub_name) not in self._df:
                    continue
                col = self._df[(full_name, sub_name)]
                if col.dtypes != np.float64:
                    col = col.str.replace(",", "").astype(np.float64)
                collected[translation[sub_name]] = col
            if len(collected) != 4:
                continue
            sub_df = pd.DataFrame([], self._df.index)
            for key in ['open', 'high', 'low', 'close']:
                sub_df[key] = collected[key]
            name = full_name.split()[0]

            symbol = Symbol(name, full_name).set_df(sub_df.dropna().copy())
            self.symbols[name] = symbol.add_log_return()


