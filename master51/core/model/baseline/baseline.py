import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from master51 import wht
from master51 import Preprocessor, Model, Reporter
from master51 import set_deterministic, make_device
from master51 import TwoAssetsScene, TwoAssetsSceneConstant
from master51.core.model.baseline.data import PairPreprocessor
from master51.core.model.baseline.generator import ModelGenerator, PairModelIndex


class BaselineReporter(Reporter):
    """
    Collect results from all possible pairs in the stock pool
    Row details:
        asset1
        asset2
        source: constant(r=0.5), iuniform, beta2, mrl, drl(ar), drl(sp)
        annual_return
        sharpe_ratio
        mdd
    """

    @property
    def sources(self):
        if not hasattr(self, "_sources"):
            self._sources = {}
        return self._sources

    @property
    def metric_list(self):
        return ["annual_return", "sharpe_ratio", "mdd", "var", "cvar", "value"]

    def collect(self):
        if hasattr(self, "_collected_cache"):
            if hasattr(self, "_collected_cache_last_len"):
                if self._collected_cache_last_len == len(self.rows):
                    return self._collected_cache

        table = {
            k: dict() for k in self.metric_list
        }
        for row in self.rows:
            cnt = self.sources.get(row["source"], 0) + 1
            self.sources[row["source"]] = cnt
            for tag in self.metric_list:
                lst = table[tag].get(row["source"], None)
                if lst is None:
                    table[tag][row["source"]] = []
                table[tag][row["source"]].append(row[tag])

        self._collected_cache = table
        self._collected_cache_last_len = len(self.rows)

        return table

    # useful analysis tools treat results as random samples

    @property
    def metrics(self) -> list:
        table = self.collect()
        return list(table.keys())

    @property
    def methods(self) -> list:
        table = self.collect()
        return list(table[self.metrics[0]].keys())

    def collect_arr(self, method: str, metric: str):
        """
        ECDF example:
        ```
        rep = benchmark.reporter
        metric = "sharpe_ratio"
        sns.ecdfplot(data={
            "half": rep.collect_arr("Constant(r=0.50)", metric),
            "MRL": rep.collect_arr("MRL[preset1225]", metric),
            "DRL": rep.collect_arr("DRL-ar[preset1225]", metric),
        })
        ```
        """
        table = self.collect()
        return np.array(table[metric][method])

    def report(self):
        results = self.collect()
        df_list = []
        for met in self.metric_list:
            df_list.append(self.get_group_df_critical(results[met]))
        df = pd.concat(df_list, axis=1, keys=self.metric_list)
        return df

    def report_threshold(self, threshold_ar=np.arange(11) * 0.05, threshold_sp=np.arange(11) * 0.2):
        results = self.collect()
        annual_return = results["annual_return"]
        sharpe_ratio = results["sharpe_ratio"]
        df_ar = self.get_group_df(annual_return, threshold_ar)
        df_sp = self.get_group_df(sharpe_ratio, threshold_sp)
        return df_ar, df_sp

    def report_quantile(self, bins=11):
        results = self.collect()
        annual_return = results["annual_return"]
        sharpe_ratio = results["sharpe_ratio"]
        df_ar = self.get_group_df_quantile(annual_return, bins)
        df_sp = self.get_group_df_quantile(sharpe_ratio, bins)
        return df_ar, df_sp

    def get_bucket(self, arr, threshold, ratio):
        greater_count = np.zeros_like(arr, dtype=int)
        for i, x in enumerate(arr):
            for t in threshold:
                if x > t:
                    greater_count[i] += 1
        bucket = np.bincount(greater_count, minlength=len(threshold)+1)
        bucket = len(arr) - np.cumsum(bucket)
        bucket = bucket[:-1]  # the last one is always zero
        if ratio:
            bucket = bucket / len(arr)
        return bucket

    def get_group_df(self, target, threshold, ratio=True):
        rows = list(target.keys())
        cols = [f"better{t:.2f}" for t in threshold]
        data = []
        for _, arr in target.items():
            data.append(self.get_bucket(arr, threshold, ratio))
        return pd.DataFrame(data, rows, cols)

    def get_group_df_quantile(self, target, bins, with_mean=False):
        prob = np.linspace(0, 1, bins, endpoint=True)
        rows = list(target.keys())

        cols = []
        # FIXME I AM DIRTY!!!
        if with_mean:
            cols.append("mean")
            cols.append("median")
        cols.extend([f"q{p:.2f}" for p in prob])

        data = []
        for _, arr in target.items():
            ent = []
            if with_mean:
                ent.append(np.mean(arr))
                ent.append(np.quantile(arr, q=0.5))
            ent.extend([np.quantile(arr, q=q) for q in prob])
            data.append(ent)
        return pd.DataFrame(data, rows, cols)

    def get_group_df_critical(self, target):
        rows = list(target.keys())

        cols = []
        cols.append("mean")
        cols.append("median")

        data = []
        for _, arr in target.items():
            ent = []
            ent.append(np.mean(arr))
            ent.append(np.quantile(arr, q=0.5))
            data.append(ent)
        return pd.DataFrame(data, rows, cols)


class BaselineModel(Model):
    """
    It's a "benchmark", the model sources manager
    """

    def get_param_default(self) -> dict:
        return {
            "pool": "top10",
            "fold": "standard",
            "tag": "test",
            "batch_size_eval": 512
        }

    @property
    def prep(self) -> PairPreprocessor:
        if not hasattr(self, "_prep"):
            raise ValueError("no default preprocessor")
        return self._prep

    def set_prep(self, prep: PairPreprocessor):
        self._prep = prep.set_pool(self("pool"))
        return self

    def set_data(self, asset1, asset2):
        self.prep.set_pair(asset1, asset2)
        self.prep.set_fold(self("fold"))
        return self

    @property
    def reporter(self) -> BaselineReporter:
        if not hasattr(self, "_reporter"):
            self._reporter = BaselineReporter()
        return self._reporter

    def set_reporter(self, reporter: BaselineReporter):
        self._reporter = reporter
        return self

    def get_pairs(self, *, reverse_unique):
        """Consider (a, b) and (b, a) as the same if `reverse_unique=True`"""
        pairs = []
        for i, a in enumerate(self.prep.stocks):
            for j, b in enumerate(self.prep.stocks):
                if reverse_unique and i > j:
                    continue
                if i != j:
                    pairs.append([a, b])
        return pairs

    def train(self, *args, **kwargs):
        self.train_constant(*args, **kwargs)

    def train_constant(self, ratios: list[float]):
        todos = self.get_pairs(reverse_unique=False)  # simulate both (a, b) and (b, a)
        with wht.get_tqdm(total=len(todos), desc="train_constant") as bar:
            for a, b in todos:
                self.set_data(a, b)
                self.simulate_constant(ratios)
                bar.update(1)
                bar.set_postfix_str(f"{a} {b} {len(self.reporter.rows)}")

    def train_beta(self, betas: list[float] | float, sample_size: int):
        if type(betas) != type([]):
            betas = [betas]

        todos = self.get_pairs(reverse_unique=True)
        with wht.get_tqdm(total=len(todos), desc="train_beta") as bar:
            for a, b in todos:
                self.set_data(a, b)
                self.simulate_beta(betas, sample_size)
                bar.update(1)
                bar.set_postfix_str(f"{a} {b} {len(self.reporter.rows)}")

    def train_model(self, generator: ModelGenerator, n_model: int, preset: str, *,
                    reverse_unique=True, callback_fn=None):
        pairs = self.get_pairs(reverse_unique=reverse_unique)
        todos = []
        for a, b in pairs:
            for idx in range(n_model):
                todos.append([a, b, idx])

        if callback_fn is None:
            callback_fn = self.get_callback_entry_collect()

        with wht.get_tqdm(total=len(todos), desc=f"{generator.__class__.__name__}") as bar:
            for ith_todo, (a, b, idx) in enumerate(todos):
                self.set_data(a, b)
                info = PairModelIndex(a, b, idx, preset)
                callback_fn(generator, self.prep, info)
                bar.update(1)
                bar.set_postfix_str(f"{a} {b} {idx}")

    def get_callback_entry_collect(self):
        def callback_fn(generator, prep, info):
            for entry in generator.collect(prep, info):
                self.reporter.append(entry)
        return callback_fn

    def get_pair_log_return(self):
        a, b = self.prep.pair
        if hasattr(self, "_cache_pair"):
            if self._cache_pair == (a, b):
                return *self._cache_pair, self._cache_lr

        # refresh cache
        dataloader = self.prep.get_dataloader(self("tag"), self("batch_size_eval"))
        packs = []
        for mini_batch in dataloader:
            _, _, lr_pair = mini_batch
            packs.append([lr_pair])
        lr_pair, = [np.concatenate(obj, axis=0) for obj in zip(*packs)]
        self._cache_pair = (a, b)
        self._cache_lr = lr_pair

        return *self._cache_pair, self._cache_lr

    def simulate_constant(self, ratios: list[float]):
        a, b, lr_pair = self.get_pair_log_return()

        for r in ratios:
            sim = TwoAssetsSceneConstant(lr_pair, r)
            self.reporter.append({
                "asset1": a,
                "asset2": b,
                "source": f"Constant(r={r:.2f})",
                **sim.result.get_profile()
            })


    def simulate_beta(self, betas: list[float], sample_size: int):
        a, b, lr_pair = self.get_pair_log_return()

        rng = np.random.RandomState(1337 + sample_size * len(self.reporter.rows))
        for i in range(sample_size):
            for j, beta in enumerate(betas):
                ratio = rng.beta(beta, beta, len(lr_pair))
                sim = TwoAssetsScene(lr_pair, ratio)
                ar = sim.result.annual_return
                sp = sim.result.sharpe
                mdd = sim.result.MDD
                self.reporter.append({
                    "asset1": a,
                    "asset2": b,
                    "source": f"Beta({beta:.2f})",
                    **sim.result.get_profile()
                })


class SimBasedReporter(BaselineReporter):
    """
    Must under a generator with matched entry
    """

    def append(self, entry):
        """
        entry_sim_based:
            asset1
            asset2
            source
            gen
            sim
        """
        sim = entry["sim"]
        value_based_entry = {
            "asset1": entry["asset1"],
            "asset2": entry["asset2"],
            "source": entry["gen"],
            "sim": entry["sim"],  # it may cause pickle file huge
            **sim.result.get_profile()
        }
        self.rows.append(value_based_entry)

    def collect_sim(self, method: str):
        sims = []
        for row in self.rows:
            if row["source"] == method:
                sims.append(row["sim"])
        return sims

    def collect_action(self, method: str):
        sims = self.collect_sim(method)
        actions = [sim.ratio_asset1 for sim in sims]
        return np.concatenate(actions, axis=0)
