from dataclasses import dataclass, field
import numpy as np
from master51.core.constants import *


@dataclass
class DailyLogReturn:
    arr: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    annual_return: float = 0
    sharpe: float = 0
    def __post_init__(self):
        arr = self.get_flatten_arr()
        if len(arr) >= 2:
            mu, sigma = self.get_annual_stats(arr)
            self.annual_return = mu
            self.sharpe = (mu - RISK_FREE_RATE) / sigma

    def get_flatten_arr(self):
        return self.arr[np.isfinite(self.arr)].reshape(-1)

    def get_annual_stats(self, arr=None):
        if arr is None:
            arr = self.get_flatten_arr()
        mu = np.mean(arr) * DAY_PER_YEAR
        sigma = np.std(arr) * DAY_PER_YEAR **.5
        return mu, sigma

    def get_profile(self) -> dict:
        return {
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe,
            "value": self.value,
            "mdd": self.MDD,
            "var": self.get_value_at_risk(),
            "cvar": self.get_conditional_value_at_risk(),
        }

    def get_value_at_risk(self, p=0.05):
        arr = self.get_flatten_arr()
        return np.quantile(arr, q=p) * DAY_PER_YEAR

    def get_conditional_value_at_risk(self, p=0.05):
        arr = self.get_flatten_arr()
        threshold = np.quantile(arr, q=p)
        return np.mean(arr[arr<threshold]) * DAY_PER_YEAR


    @property
    def worth(self):
        arr = self.get_flatten_arr()
        return np.exp(np.cumsum(arr))

    @property
    def value(self):
        """Value of the portfolio, given invested 1 at the begining"""
        return self.worth[-1]

    @property
    def drawdown(self):
        worth = self.worth
        peak = worth[0]
        output = np.zeros_like(worth)
        for i in range(len(worth)):
            if peak < worth[i]:
                peak = worth[i]
            gap = peak - worth[i]
            output[i] = gap / peak
        return output

    @property
    def MDD(self):
        return np.max(self.drawdown)

