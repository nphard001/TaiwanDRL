from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
from master51 import wht
from master51 import Preprocessor, Model, Reporter, ModelGenerator
from master51.core.model import mrl
from master51.core.model import drl
from master51.core.model import baseline
from master51.core.model.baseline import PairPreprocessor
from master51.core.model.scene import TwoAssetsScene
from master51.core.file_system import *
from master51.core.dataset.distribution import RealizedDistribution

logger = wht.get_logger()

@dataclass(frozen=True)
class PairModelIndex:
    asset1: str
    asset2: str
    ith: int  # leave the seed decision to generators
    preset: str

    @property
    def tag(self):
        return f"{self.preset}-{self.asset1}-{self.asset2}-{self.ith}"


################################################################
# Pytorch Models
################################################################

class PairRLModelGenerator(ModelGenerator):

    def get_param_preset(self, info: PairModelIndex) -> dict:
        template = {
            "fold": "standard",
            "seed": ValueError("specify seed yourself"),
            "d_model": 128,
            "l_rnn": 8,
            "d_feature": 8,
            "d_action": 5,
            "align_coef": 0.5,
            # "bin": 11,
            # "bin_width": 0.004,
            "batch_size": 32,
            "batch_size_eval": 512,
            "lr": 0.1,
            "weight_decay": 1e-8,
            "iter_train": 1000,
            "iter_cycle": 250,
            "iter_report": 250,
            "grad_multiplier": 10000
        }

        return template

    def collect(self, prep: PairPreprocessor, info: PairModelIndex) -> list:
        raise NotImplementedError("use specific class instead")

    @property
    def fcache(self):
        if not hasattr(self, "_fcache"):
            self._fcache = CACHE_M51
        return self._fcache

    def ensure_model(self, name: str):
        fcache = self.fcache
        if fcache.hit(name):
            logger.info(f"hit {name}")
            state = fcache.get(name)
            self.model.load_state_dict(state)
        else:
            logger.info(f"miss {name}")
            self.model.set_module()
            self.model.train()
            state = self.model.state_dict()
            fcache.set(state, name)

class MRLGenerator(PairRLModelGenerator):

    def get_param_preset(self, info: PairModelIndex) -> dict:
        template = super().get_param_preset(info)
        param = template.copy()

        if info.preset == "preset1225":  # sp=1.16 beat 0.85
            param["seed"] = 20221225 + info.ith
            param["align_coef"] = 0.5
            param["iter_train"] = 400
            param["iter_cycle"] = 100
            param["iter_report"] = 100
            return param

        tags = []
        if info.preset in ["paradox2000", "paradox1000", "balance500", "fast"]:
            tags.append("BackboneParadox")

        param["seed"] = 13370000 + info.ith

        if "BackboneParadox" in tags:
            param["d_action"] = 11
            param["align_coef"] = 1.0
            param["d_model"] = 256
            param["l_rnn"] = 16

        if info.preset == "paradox2000":
            param["seed"] = 20230000 + info.ith
            param["iter_train"] = 2000
            param["iter_cycle"] = 200
            param["iter_report"] = 200

        if info.preset == "paradox1000":
            param["iter_train"] = 1000
            param["iter_cycle"] = 200
            param["iter_report"] = 200

        if info.preset == "balance500":
            param["seed"] = 100 + info.ith
            param["iter_train"] = 500
            param["iter_cycle"] = 100
            param["iter_report"] = 100

        if info.preset == "fast":
            param["iter_train"] = 10
            param["iter_cycle"] = 10
            param["iter_report"] = 10

        return param

    def collect(self, prep: PairPreprocessor, info: PairModelIndex, *, use_infer=False) -> list:
        self.model = mrl.MRLModel().set_param(self.get_param_preset(info))
        self.model.set_prep(prep)

        seed = self.model("seed")
        md5_short = self.model.md5[:6]
        name = f"MRL-{info.tag}-{seed}-{md5_short}.pickle"
        name_infer = f"MRL-infer-{info.tag}-{seed}-{md5_short}.pickle"

        if self.fcache.hit(name_infer):
            logger.info(f"hit<infer> {name_infer}")
            model_infer = self.fcache.get(name_infer)
        else:
            logger.info(f"miss<infer> find model {name_infer}")
            self.ensure_model(name)
            model_infer = self.model.eval("test")
            self.fcache.set(model_infer, name_infer)

        if use_infer:
            return list(model_infer)


        dt, r2, sim = model_infer
        entry = {
            "asset1": info.asset1,
            "asset2": info.asset2,
            "source": f"MRL[{info.preset}]",
            **sim.result.get_profile()
        }
        return [entry]


class DRLGenerator(PairRLModelGenerator):

    def get_param_preset(self, info: PairModelIndex) -> dict:
        template = super().get_param_preset(info)
        param = template.copy()
        param["bin"] = 11
        param["bin_width"] = 0.004

        if info.preset == "preset1225":  # sp=1.049 beat 0.85
            param["seed"] = 20221225 + info.ith
            param["align_coef"] = 0.9
            param["iter_train"] = 1000
            param["iter_cycle"] = 100
            param["iter_report"] = 100
            return param

        tags = ["C21"]
        if info.preset in ["paradox2000", "paradox1000", "balance500", "fast"]:
            tags.append("BackboneParadox")

        param["seed"] = 13370000 + info.ith

        if "C21" in tags:
            param["bin"] = 21
            param["bin_width"] = 0.0025

        if "BackboneParadox" in tags:
            param["d_action"] = 11
            param["align_coef"] = 1.0
            param["d_model"] = 256
            param["l_rnn"] = 16

        if info.preset == "paradox2000":
            param["seed"] = 20230000 + info.ith
            param["iter_train"] = 2000
            param["iter_cycle"] = 200
            param["iter_report"] = 200

        if info.preset == "paradox1000":
            param["iter_train"] = 1000
            param["iter_cycle"] = 200
            param["iter_report"] = 200

        if info.preset == "balance500":
            param["seed"] = 100 + info.ith
            param["iter_train"] = 500
            param["iter_cycle"] = 100
            param["iter_report"] = 100

        if info.preset == "fast":
            param["iter_train"] = 10
            param["iter_cycle"] = 10
            param["iter_report"] = 10

        return param

    def collect(self, prep: PairPreprocessor, info: PairModelIndex, *, use_infer=False) -> list:
        self.model = drl.DRLModel().set_param(self.get_param_preset(info))
        self.model.set_prep(prep)

        seed = self.model("seed")
        md5_short = self.model.md5[:6]
        name = f"DRL-{info.tag}-{seed}-{md5_short}.pickle"
        name_infer = f"DRL-infer-{info.tag}-{seed}-{md5_short}.pickle"

        if self.fcache.hit(name_infer):
            logger.info(f"hit<infer> {name_infer}")
            model_infer = self.fcache.get(name_infer)
        else:
            logger.info(f"miss<infer> find model {name_infer}")
            self.ensure_model(name)
            model_infer = self.model.eval("test", use_prob=True)
            self.fcache.set(model_infer, name_infer)

        if use_infer:
            return list(model_infer)

        dt, entropy, prob, sim1, sim2 = model_infer
        entries = []
        for name, sim in zip(["sp", "ar"], [sim1, sim2]):
            entry = {
                "asset1": info.asset1,
                "asset2": info.asset2,
                "source": f"DRL-{name}[{info.preset}]",
                **sim.result.get_profile()
            }
            entries.append(entry)
        return entries

    def collect_prob(self, prep: PairPreprocessor, info: PairModelIndex):
        # [dt[0], dt[-1]], entropy, prob, sim1, sim2
        model_infer = self.collect(prep, info, use_infer=True)
        prob: np.ndarray = model_infer[2]
        sim: TwoAssetsScene = model_infer[-1]
        return prob, sim

    def collect_dist(self, prep: PairPreprocessor, info: PairModelIndex, *, resolution: int):
        self.model = drl.DRLModel().set_param(self.get_param_preset(info))
        self.model.set_prep(prep)
        seed = self.model("seed")
        md5_short = self.model.md5[:6]
        name_dist = f"DRL-dist-sample{resolution}-{info.tag}-{seed}-{md5_short}.pickle"

        if self.fcache.hit(name_dist):
            logger.info(f"hit<dist> {name_dist}")
            dist_state = self.fcache.get(name_dist)
            dist, prob, sim = dist_state
        else:
            logger.info(f"miss<dist> compute {name_dist}")
            prob, sim = self.collect_prob(prep, info)
            dist = self.model.dirac.to_dist(prob, resolution=resolution)
            dist_state = dist, prob, sim
            self.fcache.set(dist_state, name_dist)

        return dist, prob, sim


################################################################
# Non-Pytorch Models
################################################################

class PairCPUModelGenerator(ModelGenerator):

    def collect(self, prep: PairPreprocessor, info: PairModelIndex) -> list:
        raise NotImplementedError("use specific class instead")

    def train(self):
        raise NotImplementedError("define what to do if model cache miss")

    @property
    def fcache(self):
        if not hasattr(self, "_fcache"):
            self._fcache = CACHE_M51
        return self._fcache

    def ensure_model(self, name: str):
        fcache = self.fcache
        if fcache.hit(name):
            logger.info(f"hit {name}")
            self.model = fcache.get(name)
        else:
            logger.info(f"miss {name}")
            self.train()
            fcache.set(self.model, name)

    # FIXME I am dirty COPY from MRL
    @property
    def collate_fn(self):
        if not hasattr(self, "_collate_fn"):
            self._collate_fn = self.get_collate_fn_default()
        return self._collate_fn

    def get_collate_fn_default(self):
        def collate_fn(batch):
            dt, xarr, lr_pair = list(zip(*batch))
            dt = np.array(dt, dtype=object)
            xarr = np.stack(xarr, axis=0)
            lr_pair = np.stack(lr_pair, axis=0)
            xtns = torch.as_tensor(xarr, dtype=torch.float32)
            lr_pair = torch.as_tensor(lr_pair, dtype=torch.float32)
            return dt, xtns, lr_pair
        return collate_fn


class ConstantModelGenerator(PairCPUModelGenerator):

    def __init__(self, c: float):
        self.c = c

    def collect(self, prep: PairPreprocessor, info: PairModelIndex, *, use_infer: bool) -> list:
        self.prep = prep

        name = f"const{self.c:.2f}-{info.asset1}-{info.asset2}.pickle"
        self.ensure_model(name)

        if use_infer:
            return [self.c, self.model["sim"]]
        raise NotImplementedError("use_infer must be True under CPU-based models")


    def train(self):
        self.prep.set_collate_fn(self.collate_fn)
        dataloader = self.prep.get_dataloader("test", 512)
        packs = []
        for mini_batch in dataloader:
            _, _, lr_pair = mini_batch
            packs.append([lr_pair])
        lr_pair, = [np.concatenate(obj, axis=0) for obj in zip(*packs)]

        alphas = np.ones(lr_pair.shape[0], dtype=float) * self.c
        self.model = {"sim": TwoAssetsScene(lr_pair, alphas)}


################################################################
# Generator of generators
################################################################

class Sim900Generator(ModelGenerator):

    def __init__(self):
        self.generators = {
            "Buy & Hold": ConstantModelGenerator(1.0),
            "Fifty-Fifty": ConstantModelGenerator(0.5),
            "MRL": MRLGenerator(),
            "DRL": DRLGenerator()
        }

    def collect(self, prep: PairPreprocessor, info: PairModelIndex) -> list:
        """
        Always use_infer=True, and assume last one is the candidate_sim
        """
        entries = []
        for gen_name, gen in self.generators.items():
            model_infer = gen.collect(prep, info, use_infer=True)
            candidate_sim: TwoAssetsScene = model_infer[-1]
            entry = {
                "asset1": info.asset1,
                "asset2": info.asset2,
                "source": f"{gen_name}[{info.preset}]",
                "gen": f"{gen_name}",
                "sim": candidate_sim
            }
            entries.append(entry)
        return entries

class RealizedDistortionDiscreteDecisionFunction:
    def __init__(self, action_bins: int):
        self.actions = np.linspace(0, 1, action_bins)
    def __call__(self, action_to_dist: list[RealizedDistribution]) -> float:
        action_score = [self.score(dist) for dist in action_to_dist]
        return self.actions[np.argmax(action_score)]
    def score(self, dist: RealizedDistribution) -> float:
        return dist.mean

class MeanDistortion(RealizedDistortionDiscreteDecisionFunction):
    def score(self, dist: RealizedDistribution):
        return dist.mean

class SharpeDistortion(RealizedDistortionDiscreteDecisionFunction):
    def score(self, dist: RealizedDistribution):
        return dist.mean / dist.std

class QuantileDistortion(RealizedDistortionDiscreteDecisionFunction):
    def __init__(self, action_bins: int, p: float):
        self.actions = np.linspace(0, 1, action_bins)
        self.p = p
    def score(self, dist: RealizedDistribution):
        return dist(self.p)

class ConditionalDistortion(QuantileDistortion):
    def score(self, dist: RealizedDistribution):
        return dist.condition(upper_q=self.p).mean


class DistortionGenerator(ModelGenerator):
    """
    Make different decisions under a same probability measure
    """

    def __init__(self, gen_prob=None, preset="DRL", resolution=10):
        if gen_prob is None:
            if preset == "DRL":
                gen_prob = DRLGenerator()
            if preset == "Fast":
                gen_prob = DRLGenerator()
        self.preset = preset
        self.gen_prob: DRLGenerator = gen_prob
        self.resolution = resolution

        bins = 11
        self.generators: dict[str, RealizedDistortionDiscreteDecisionFunction] = {
            # "DRL": MeanDistortion(bins),
            "DRL": None,
            "DRL-SR": SharpeDistortion(bins),
            "DRL-Median": QuantileDistortion(bins, 0.5),
            "DRL-RiskSeeking": QuantileDistortion(bins, 0.95),
            "DRL-VaR": QuantileDistortion(bins, 0.05),
            "DRL-CVaR": ConditionalDistortion(bins, 0.05),
        }
        if preset == "Fast":
            self.generators = {
                # "DRL": MeanDistortion(bins),
                "DRL": None,
                "DRL-SR": SharpeDistortion(bins),
            }

    def collect(self, prep: PairPreprocessor, info: PairModelIndex) -> list:
        dist, prob, sim0 = self.gen_prob.collect_dist(prep, info, resolution=self.resolution)
        entries = []
        for gen_name, gen in self.generators.items():
            alphas = sim0.ratio_asset1.copy()
            if gen is not None:
                for i in range(len(alphas)):
                    dists: list[RealizedDistribution] = dist[i]
                    alphas[i] = gen(dists)
            sim = TwoAssetsScene(sim0.pair_tomorrow, alphas)
            entry = {
                "asset1": info.asset1,
                "asset2": info.asset2,
                "source": f"{gen_name}[{info.preset}]",
                "gen": f"{gen_name}",
                "sim": sim
            }
            entries.append(entry)
        return entries


