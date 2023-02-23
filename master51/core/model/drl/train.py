from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
from torch import nn
import torch.nn.functional as F
from master51 import wht
from master51 import Preprocessor, Model, Reporter
from master51 import set_deterministic, make_device
from master51 import TwoAssetsScene
from master51 import RNNFeatureEncoder, SequenceEdgeMeanPooling, GroupLinear, MSERegressionStabilizer, LOG_RETURN_MU, LOG_RETURN_SIGMA
from master51.core.model import mrl
from master51.core.dataset.distribution import Dirac


class DRLModule(nn.Module):
    """
    input: (batch, seq_len, d_feature)
    output: (batch, d_action, d_bin)
    """

    def __init__(self, *, d_model, l_rnn, d_feature, d_action, d_bin, align_coef):
        nn.Module.__init__(self)
        self.rnn = RNNFeatureEncoder(d_model=d_model, l_rnn=l_rnn, d_feature=d_feature)
        self.output = GroupLinear(d_model, d_bin, d_action, align_coef)
        self.output_shape = (-1, d_action, d_bin)

    def forward(self, input):
        tns = self.rnn(input)     # (batch, d_model)
        tns = self.output(tns)    # (batch, d_action, d_bin)
        logits = tns.reshape(*self.output_shape)
        return logits


class DRLReporter(Reporter):

    def report(self, data):
        row = [data["iter_done"]]
        for tag in ["train", "valid", "test"]:
            dt, cross_entropy, sim1, sim2 = data[tag]
            ppx = np.exp(cross_entropy).reshape(-1)
            sp1 = sim1.result.sharpe
            sp2 = sim2.result.sharpe
            row.extend([np.mean(ppx), sp1, sp2])
        self.append(row)
        print(self.get_df().tail(1))

    def get_df(self):
        columns = ["iter_done"]
        for ds in ["ds1", "ds2", "ds3"]:
            columns.extend([
                f"{ds}_ppx",
                f"{ds}_spSig",
                f"{ds}_spMu",])
        df = pd.DataFrame(self._rows, None, columns)
        return df


class DRLModel(mrl.MRLModel):
    def get_param_default(self) -> dict:
        return {
            "fold": "standard",
            "seed": 1337000,
            "d_model": 128,
            "l_rnn": 8,
            "d_feature": 8,
            "d_action": 5,
            "align_coef": 0.5,
            "bin": 11,
            "bin_width": 0.004,
            "batch_size": 32,
            "batch_size_eval": 512,
            "lr": 0.1,
            "weight_decay": 1e-8,
            "iter_train": 2000,
            "iter_cycle": 500,
            "iter_report": 500,
            "grad_multiplier": 10000
        }

    def get_func_init(self):
        set_deterministic(self("seed"))
        return DRLModule(
            d_model=self("d_model"),
            l_rnn=self("l_rnn"),
            d_feature=self("d_feature"),
            d_action=self("d_action"),
            d_bin=self("bin"),
            align_coef=self("align_coef"))

    def get_reporter_default(self):
        return DRLReporter()

    @property
    def dirac(self) -> Dirac:
        if not hasattr(self, "_dirac"):
            self._dirac = Dirac(self("bin"), self("bin_width"))
        return self._dirac

    def get_collate_fn_default(self):
        def collate_fn(batch):
            dt, xarr, lr_pair = list(zip(*batch))
            dt = np.array(dt, dtype=object)
            xarr = np.stack(xarr, axis=0)
            lr_pair = np.stack(lr_pair, axis=0)
            lr_action = np.stack([r * lr_pair[:, 0] + (1-r) * lr_pair[:, 1]
                for r in self.get_action_ratio()], axis=1)
            ctg_action = self.dirac.to_ctg(lr_action)
            xtns = torch.as_tensor(xarr, dtype=torch.float32)
            lr_pair = torch.as_tensor(lr_pair, dtype=torch.float32)
            lr_action = torch.as_tensor(lr_action, dtype=torch.float32)
            ctg_action = torch.as_tensor(ctg_action, dtype=torch.long)
            return dt, xtns, lr_pair, lr_action, ctg_action
        return collate_fn

    def train_step(self, mini_batch):
        GPU, CPU, _ = self.get_device_fn()
        dt, xtns, lr_pair, lr_action, ctg_action = mini_batch
        xtns, lr_pair, lr_action, ctg_action = GPU(xtns, lr_pair, lr_action, ctg_action)

        self.func.train()
        self.optim.zero_grad()
        logits = self.func(xtns)
        logits_f = logits.reshape(-1, self("bin"))
        ctg_action_f = ctg_action.reshape(-1)
        entropy = F.cross_entropy(logits_f, ctg_action_f, reduction="none")  # (B*5, 11)
        loss = entropy.mean() * self("grad_multiplier")
        loss.backward()

        p = self.func.parameters()
        g = nn.utils.clip_grad_norm_(p, 1.0)

        self.optim.step()
        self.sched.step()

    def eval(self, tag, *, use_prob=False):
        dataloader = self.prep.get_dataloader(tag, self("batch_size_eval"))
        self.ensure_gpu()

        packs = []
        for mini_batch in dataloader:
            packs.append(self.eval_step(mini_batch))
        dt, prob, mu, sigma, entropy, lr_pair, lr_action, ctg_action = [np.concatenate(obj, axis=0) for obj in zip(*packs)]

        action = np.argmax(mu/sigma, axis=1)
        # action = np.argmax(lr_action, axis=1)
        ratio = np.choose(action, self.get_action_ratio())
        sim1 = TwoAssetsScene(lr_pair, ratio)

        action = np.argmax(mu, axis=1)
        ratio = np.choose(action, self.get_action_ratio())
        sim2 = TwoAssetsScene(lr_pair, ratio)

        if use_prob:
            return [dt[0], dt[-1]], entropy, prob, sim1, sim2
        return [dt[0], dt[-1]], entropy, sim1, sim2

    def eval_step(self, mini_batch):
        GPU, CPU, _ = self.get_device_fn()
        dt, xtns, lr_pair, lr_action, ctg_action = mini_batch
        xtns, lr_pair, lr_action, ctg_action = GPU(xtns, lr_pair, lr_action, ctg_action)

        self.func.eval()
        with torch.no_grad():
            logits = self.func(xtns)
            logits_f = logits.reshape(-1, self("bin"))
            ctg_action_f = ctg_action.reshape(-1)
            entropy = F.cross_entropy(logits_f, ctg_action_f, reduction="none")  # (B*5, 11)
            entropy = entropy.reshape(-1, self("d_action"))  # (B, 5)
        prob = CPU(logits.softmax(dim=-1))
        mu, sigma = self.dirac.to_stats(prob)

        return [dt[:, 1], prob, mu, sigma, *CPU(entropy, lr_pair, lr_action, ctg_action)]
