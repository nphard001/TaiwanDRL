import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from master51 import wht
from master51 import Preprocessor, TEJParser, SymbolSet
from master51 import FULL2951, TOP10, TOP5, CHERRY3, CHERRY2


class PairFutureDataset(Dataset):
    """Generalized form of mrl.ZFutureDataset"""

    def __init__(self, df, window_size):
        self._df = df.copy()
        self.window_size = window_size
        self.valid_iloc = []
        self.label_iloc = []
        self.feature_iloc = []

        for i, (_, col) in enumerate(self._df.columns):
            if col == "future":
                self.label_iloc.append(i)
            else:
                self.feature_iloc.append(i)

        # NOTE: all data must be finite, future inference IRL is not included
        for i in range(self._df.shape[0]):
            ed = i + 1
            st = ed - window_size
            if st < 0:
                continue  # ignore small window
            total = window_size * self._df.shape[1]
            ok = np.sum(np.isfinite(self._df.values[st:ed]))
            if ok == total:
                self.valid_iloc.append([i, st, ed])

    def __len__(self):
        return len(self.valid_iloc)

    def __getitem__(self, idx):
        idx_df, st, ed = self.valid_iloc[idx]
        sdf = self._df.iloc[st:ed]
        dates = [sdf.index[0], sdf.index[-1]]
        xarr = sdf.values[:, self.feature_iloc]
        yarr = sdf.values[-1, self.label_iloc]
        return dates, xarr, yarr

    @classmethod
    def collate_fn(cls, batch):
        """
        Default collate function
        It's numpy format, convert to tensor by yourself if needed
        """
        dt, xarr, yarr = list(zip(*batch))
        dt = np.array(dt, dtype=object)
        xarr = np.stack(xarr, axis=0)
        yarr = np.stack(yarr, axis=0)
        return dt, xarr, yarr


class PairPreprocessor(Preprocessor):
    """Generalized form of mrl.DemoPreprocessor"""

    def __init__(self, *, row_window_size=10, z_window_size=60):
        self.row_window_size = row_window_size
        self.z_window_size = z_window_size
        self.pool_name: str | None = None

    @property
    def stocks(self) -> list[str]:
        if self.pool_name == "cherry2":
            return CHERRY2
        if self.pool_name == "cherry3":
            return CHERRY3
        if self.pool_name == "cherry":
            return CHERRY2
        if self.pool_name == "top10":
            return TOP10
        if self.pool_name == "top5":
            return TOP5
        return FULL2951

    @property
    def major(self) -> SymbolSet:
        if not hasattr(self, "_major"):
            env = wht.PathEnv().set_root()
            env.base_path = env('dat', 'master51')
            raw = TEJParser(env('tw50.csv'))

            major = SymbolSet(raw.symbols)
            major = major.build_subset(self.stocks)

            for k in major:
                sym = major[k]
                sym.add_future()
                sym.add_zprice(window_size=self.z_window_size)
            self._major = major
        return self._major

    def set_pool(self, name):
        self.pool_name = name
        return self

    def set_pair(self, asset1, asset2):
        self.pair = [asset1, asset2]
        symbol_subset = self.major.build_subset(self.pair)
        pair_df = symbol_subset.collect(["zopen", "zhigh", "zlow", "zclose", "future"])
        pair_df = pair_df.dropna()
        self.pair_dataset = PairFutureDataset(pair_df, window_size=self.row_window_size)
        return self

    def get_pair_full_name(self):
        return [self.major.symbols[name].full_name for name in self.pair]

    def set_fold(self, name):
        self.fold = {}
        dataset = self.pair_dataset
        if name == "optim":
            n0 = 1900
            n1 = 100  # very small training set make sure model works in-sample
            n2 = 250
            n3 = 500
        elif name == "medium":
            n0 = 1500
            n1 = 500
            n2 = 250
            n3 = 500
        elif name == "standard":
            n0 = 0
            n1 = 2000
            n2 = 250
            n3 = 500
        else:
            raise KeyError(f"no such fold {name}")
        self.fold["train"] = Subset(dataset, range(n0, n0+n1))
        self.fold["valid"] = Subset(dataset, range(n0+n1, n0+n1+n2))
        self.fold["test"] = Subset(dataset, range(n0+n1+n2, n0+n1+n2+n3))
        return self

    def get_dataloader(self, tag, batch_size, *, num_workers=None, train=False):
        if num_workers is None:
            if os.name == "nt":
                num_workers = 0
            else:
                num_workers = os.cpu_count()

        dataset = self.fold[tag]
        cfn = self.collate_fn
        return DataLoader(dataset, batch_size, num_workers=num_workers,
            shuffle=train, drop_last=train, collate_fn=cfn)

    @property
    def collate_fn(self):
        if not hasattr(self, "_collate_fn"):
            self._collate_fn = self.get_collate_fn_default()
        return self._collate_fn

    def set_collate_fn(self, cfn):
        self._collate_fn = cfn
        return self

    def get_collate_fn_default(self):
        return PairFutureDataset.collate_fn
