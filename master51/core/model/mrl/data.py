import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from master51 import wht
from master51 import Preprocessor, TEJParser, SymbolSet

class ZFutureDataset(Dataset):

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
        dt, xarr, yarr = list(zip(*batch))
        dt = np.array(dt, dtype=object)
        xarr = np.stack(xarr, axis=0)
        yarr = np.stack(yarr, axis=0)
        xtns = torch.as_tensor(xarr, dtype=torch.float32)
        ytns = torch.as_tensor(yarr, dtype=torch.float32)
        return dt, xtns, ytns


class DemoPreprocessor(Preprocessor):

    def get(self, name):
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache.get(name, None)

    def load(self):
        major = self.get("major")
        if major is None:
            env = wht.PathEnv().set_root()
            env.base_path = env('dat', 'master51')
            raw = TEJParser(env('tw50.csv'))

            major = SymbolSet(raw.symbols)
            self._cache["major"] = major

        demo_df = self.get("demo_df")
        if demo_df is None:
            demo_pair = major.build_subset(["2330", "2884"])
            for k in demo_pair:
                demo_pair[k].add_future()
                demo_pair[k].add_zprice(window_size=60)
            complete = demo_pair.collect(["future", "zclose"])
            complete_index = complete.dropna().index
            complete_set = demo_pair.collect(["zopen", "zhigh", "zlow", "zclose", "future"])
            demo_df = complete_set.loc[complete_index]
            self._cache["demo_df"] = demo_df

        demo_dataset = self.get("demo_dataset")
        if demo_dataset is None:
            demo_dataset = ZFutureDataset(demo_df, window_size=10)
            self._cache["demo_dataset"] = demo_dataset
        return self

    def set_fold(self, name):
        self.fold = {}
        dataset = self.get("demo_dataset")
        if name == "optim":
            n0 = 1900
            n1 = 100  # very small training set make sure model works in-sample
            n2 = 250
            n3 = 500
            self.fold["train"] = Subset(dataset, range(n0, n0+n1))
            self.fold["valid"] = Subset(dataset, range(n0+n1, n0+n1+n2))
            self.fold["test"] = Subset(dataset, range(n0+n1+n2, n0+n1+n2+n3))
        elif name == "standard":
            n1 = 2000
            n2 = 250
            n3 = 500
            self.fold["train"] = Subset(dataset, range(0, n1))
            self.fold["valid"] = Subset(dataset, range(n1, n1+n2))
            self.fold["test"] = Subset(dataset, range(n1+n2, n1+n2+n3))
        else:
            raise KeyError(f"no such fold {name}")
        return self

    def get_dataloader(self, tag, batch_size, *, num_workers=0, train=False):
        dataset = self.fold[tag]
        cfn = self.get_collate_fn()
        return DataLoader(dataset, batch_size, num_workers=num_workers,
            shuffle=train, drop_last=train, collate_fn=cfn)

    def get_collate_fn(self):
        if not hasattr(self, "collate_fn"):
            self.collate_fn = ZFutureDataset.collate_fn
        return self.collate_fn

    def set_collate_fn(self, cfn):
        self.collate_fn = cfn
        return self
