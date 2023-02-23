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
# from master51.core.model.mrl.data import DemoPreprocessor

class MRLModule(nn.Module):
    """
    input: (batch, seq_len, d_feature)
    output: (batch, d_action)
    """

    def __init__(self, *, d_model, l_rnn, d_feature, d_action, align_coef):
        nn.Module.__init__(self)
        self.rnn = RNNFeatureEncoder(d_model=d_model, l_rnn=l_rnn, d_feature=d_feature)
        self.output = GroupLinear(d_model, 1, d_action, align_coef)
        self.regression = MSERegressionStabilizer(LOG_RETURN_MU, LOG_RETURN_SIGMA)

    def forward(self, input):
        tns = self.rnn(input)     # (batch, d_model)
        tns = self.output(tns)    # (batch, d_action, 1)
        tns = tns.squeeze(dim=2)  # (batch, d_action)
        tns = self.regression(tns)
        return tns

    def get_loss(self, y_pred, y_true):
        loss = self.regression.get_mse(y_pred, y_true)
        return loss


class MRLReporter(Reporter):

    def report(self, data):
        row = [data["iter_done"]]
        for tag in ["train", "valid", "test"]:
            dt, r2, sim = data[tag]
            r2_min = np.min(r2)
            r2_max = np.max(r2)
            ar = sim.result.annual_return
            sp = sim.result.sharpe
            row.extend([r2_min, r2_max, ar, sp])
        self.append(row)
        print(self.get_df().tail(1))

    def get_df(self):
        columns = ["iter_done"]
        for ds in ["ds1", "ds2", "ds3"]:
            columns.extend([
                f"{ds}_p",
                f"{ds}_q",
                f"{ds}_ar",
                f"{ds}_sp",])
        df = pd.DataFrame(self._rows, None, columns)
        return df


class MRLModel(Model):
    def get_param_default(self) -> dict:
        return {
            "fold": "standard",
            "seed": 1337000,
            "d_model": 128,
            "l_rnn": 8,
            "d_feature": 8,
            "d_action": 5,
            "align_coef": 0.5,
            "batch_size": 32,
            "batch_size_eval": 512,
            "lr": 0.1,
            "weight_decay": 1e-8,
            "iter_train": 2000,
            "iter_cycle": 500,
            "iter_report": 500,
            "grad_multiplier": 10000
        }

    def state_dict(self) -> dict:
        self.ensure_cpu()  # FIXME I am dirty, a state function with side-effects
        return {
            "param": self.param,
            "func": self.func.state_dict(),
            "reporter": self.reporter.state_dict()
        }

    def load_state_dict(self, state_dict):
        self._param = state_dict["param"]

        self.func = self.get_func_init()
        self.func.load_state_dict(state_dict["func"])

        # FIXME I am buggy, won't work in all the general cases
        self.reporter.load_state_dict(state_dict["reporter"])

    def set_prep(self, prep: Preprocessor):
        self.prep = prep.set_fold(self("fold"))
        self.prep.set_collate_fn(self.collate_fn)
        return self

    @property
    def reporter(self) -> Reporter:
        if not hasattr(self, "_reporter"):
            self._reporter = self.get_reporter_default()
        return self._reporter

    def get_reporter_default(self):
        return MRLReporter()

    def set_reporter(self, reporter: Reporter):
        self._reporter = reporter
        return self

    def get_func_init(self):
        set_deterministic(self("seed"))
        return MRLModule(
            d_model=self("d_model"),
            l_rnn=self("l_rnn"),
            d_feature=self("d_feature"),
            d_action=self("d_action"),
            align_coef=self("align_coef"))

    def set_module(self):
        self.func = self.get_func_init()
        return self

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

    def get_device_fn(self):
        if not hasattr(self, "device_fn"):
            self.device_fn = make_device()
        return self.device_fn

    @property
    def device(self):
        if not hasattr(self, "_device"):
            self._device = "cpu"
        return self._device

    def ensure_cpu(self):
        if self.device == "cpu":
            return
        self.func.to("cpu")
        self._device = "cpu"

    def ensure_gpu(self):
        if self.device == "gpu":
            return
        _, _, device_name = self.get_device_fn()
        self.func.to(device_name)
        self._device = "gpu"
        self.prep.set_collate_fn(self.collate_fn)  # NOTE not sure when cfn should be ready

    def get_action_ratio(self):
        return np.linspace(0, 1, self("d_action"), endpoint=True)

    def train(self):
        lr = self("lr")
        self.optim = torch.optim.SGD(
            self.func.parameters(), lr, weight_decay=self("weight_decay")*self("grad_multiplier"),
            momentum=0.9, nesterov=True)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim, self("iter_cycle"), eta_min=lr*0.001)
        iter_train = self("iter_train")
        iter_done = 0

        dataloader = self.prep.get_dataloader("train", self("batch_size"), train=True)
        self.ensure_gpu()

        with wht.misc.get_tqdm(total=iter_train) as bar:
            bar.set_description_str(f"{self.prep.get_pair_full_name()} seed{self('seed')}")
            while iter_done < iter_train:
                for mini_batch in dataloader:
                    self.train_step(mini_batch)
                    iter_done += 1
                    lr_now = self.sched.get_last_lr()[0]
                    epoch = iter_done * self("batch_size") / len(dataloader)
                    bar.update(1)
                    bar.set_postfix_str(f"epoch={epoch:.2f} lr={lr_now:.8f}")
                    if iter_done % self("iter_report") == 0:
                        self.reporter.report({
                            "iter_done": iter_done,
                            "train": self.eval("train"),
                            "valid": self.eval("valid"),
                            "test": self.eval("test"),
                        })
                    if iter_done >= iter_train:
                        break


    def train_step(self, mini_batch):
        GPU, CPU, _ = self.get_device_fn()
        dt, xtns, ytns = mini_batch
        xtns, ytns = GPU(xtns, ytns)

        y_true = torch.stack([r * ytns[:, 0] + (1-r) * ytns[:, 1]
            for r in self.get_action_ratio()], dim=1)

        self.func.train()
        self.optim.zero_grad()
        y_pred = self.func(xtns)
        loss = self.func.get_loss(y_pred, y_true).sum() * self("grad_multiplier")
        loss.backward()

        p = self.func.parameters()
        g = nn.utils.clip_grad_norm_(p, 1.0)

        self.optim.step()
        self.sched.step()

    def eval(self, tag):
        dataloader = self.prep.get_dataloader(tag, self("batch_size_eval"))
        self.ensure_gpu()

        packs = []
        for mini_batch in dataloader:
            packs.append(self.eval_step(mini_batch))
        dt, y_pred, y_true, pair_lr = [np.concatenate(obj, axis=0) for obj in zip(*packs)]

        r2 = np.stack([r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_pred.shape[1])], 0)
        action = np.argmax(y_pred, axis=1)
        # action = np.argmax(y_true, axis=1)
        ratio = np.choose(action, self.get_action_ratio())
        sim = TwoAssetsScene(pair_lr, ratio)
        return [dt[0], dt[-1]], r2, sim

    def eval_step(self, mini_batch):
        GPU, CPU, _ = self.get_device_fn()
        dt, xtns, ytns = mini_batch
        xtns, ytns = GPU(xtns, ytns)

        y_true = torch.stack([r * ytns[:, 0] + (1-r) * ytns[:, 1]
            for r in self.get_action_ratio()], dim=1)

        self.func.eval()
        with torch.no_grad():
            y_pred = self.func(xtns)
        return [dt[:, 1], CPU(y_pred), CPU(y_true), CPU(ytns)]
