from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from master51 import wht


class Distribution(ABC):

    @property
    def mean(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def condition(self, *, lower=None, upper=None) -> Distribution:
        """Conditional distribution, a subset"""


class RealizedDistribution(Distribution):
    """Simply put: treat a sample as a distribution"""

    def __init__(self, samples):
        self.arr = np.sort(np.array(samples).copy().reshape(-1))

    def __call__(self, q):
        """Syntax sugar for Inversed CDF, a super useful function, i.e. quantile"""
        return np.quantile(self.arr, q=q)

    @property
    def mean(self) -> float:
        return np.mean(self.arr)

    @property
    def std(self) -> float:
        return np.std(self.arr)

    def condition(self, *, lower_q=None, upper_q=None) -> RealizedDistribution:
        arr_sub = self.arr.copy()
        if lower_q is not None:
            lower = self(lower_q)
            arr_sub = arr_sub[arr_sub>=lower]
        if upper_q is not None:
            upper = self(upper_q)
            arr_sub = arr_sub[arr_sub<=upper]
        return RealizedDistribution(arr_sub)



@dataclass
class Dirac:
    num: int
    width: float
    def __post_init__(self):
        self.values = np.array([(i - (self.num-1)/2) * self.width for i in range(self.num)])

    def to_ctg(self, arr):
        shape = arr.shape
        distance = arr.copy().reshape(-1, 1)
        distance = np.repeat(distance, self.num, axis=1)
        distance = np.abs(distance - self.values)
        ctg = np.argmin(distance, axis=1)
        ctg = ctg.reshape(shape)
        return ctg

    def to_stats(self, arr):
        shape = arr.shape[:-1]
        prob = arr.copy().reshape(-1, self.num)
        mu = np.sum(prob * self.values, axis=1)
        mu_f = mu.copy().reshape(-1, 1)
        mu_f = np.repeat(mu_f, self.num, axis=1)
        sigma = np.sum(prob * (self.values - mu_f) ** 2, axis=1) ** 0.5
        return mu.reshape(shape), sigma.reshape(shape)

    def to_dist(self, prob: np.ndarray, resolution: int):
        return self.to_dist_by_sample(prob, resolution)

    def to_dist_by_sample(self, prob: np.ndarray, resolution: int):
        assert prob.shape[-1] == self.num  # prob(500, 11, 21) <---> C=21 -> dist(500, 11)
        rng = np.random.RandomState(1337)
        prob_flatten = prob.copy().reshape(-1, self.num)
        output_shape = prob.shape[:-1]
        output = np.empty(prob_flatten.shape[0], dtype=object)
        for i in range(output.shape[0]):
            P = prob_flatten[i]
            samples = rng.choice(self.values, resolution, p=P)
            output[i] = RealizedDistribution(samples)
        return output.reshape(output_shape)

    def to_dist_by_dup(self, prob: np.ndarray, resolution: int):
        assert prob.shape[-1] == self.num  # prob(500, 11, 21) <---> C=21 -> dist(500, 11)
        prob_flatten = prob.copy().reshape(-1, self.num)
        output_shape = prob.shape[:-1]
        output = np.empty(prob_flatten.shape[0], dtype=object)
        for i in range(output.shape[0]):
            P = prob_flatten[i]
            samples_mem = np.zeros((self.num+5) * (resolution+5), dtype=float)
            samples_cnt = 0
            for j in range(self.num):
                zj = self.values[j]
                pj = P[j]
                n = int(np.ceil(pj * resolution))
                for _ in range(n):
                    samples_mem[samples_cnt] = zj
                    samples_cnt += 1
            output[i] = RealizedDistribution(samples_mem[:samples_cnt])
        return output.reshape(output_shape)

