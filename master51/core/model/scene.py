"""
Model Evaluation Scenes
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from master51 import wht
from master51.core.dataset.functional import DailyLogReturn


@dataclass
class TwoAssetsScene:
    pair_tomorrow: np.ndarray = field(default_factory=list, repr=False)
    ratio_asset1: np.ndarray = field(default_factory=list, repr=False)
    result: DailyLogReturn = field(default_factory=lambda: DailyLogReturn())

    def __post_init__(self):
        assert self.pair_tomorrow.shape[0] == self.ratio_asset1.shape[0]
        N = self.ratio_asset1.shape[0]
        result = []
        for i in range(N):
            r1 = self.ratio_asset1[i]
            r2 = 1 - r1
            result.append(r1 * self.pair_tomorrow[i, 0] + r2 * self.pair_tomorrow[i, 1])
        result = np.array(result)
        self.result = DailyLogReturn(result)


@dataclass
class TwoAssetsSceneConstant:
    pair_tomorrow: np.ndarray = field(default_factory=list, repr=False)
    const_ratio_asset1: float = 0.5
    result: DailyLogReturn = field(default_factory=lambda: DailyLogReturn())

    def __post_init__(self):
        ratio = np.ones_like(self.pair_tomorrow[:, 0]) * self.const_ratio_asset1
        scene = TwoAssetsScene(self.pair_tomorrow, ratio)
        self.result = scene.result
