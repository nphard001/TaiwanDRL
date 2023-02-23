"""
Economics / Financial useful
"""
import numpy as np
import scipy.stats


def crra(arr, eta=1.0, eps=1e-12):
    """Constant Relative Risk Aversion (CRRA) utility function"""
    arr = np.clip(arr, eps, np.inf)
    if np.abs(1-eta)<eps:
        return np.log(arr)
    return (np.power(arr, 1-eta)-1)/(1-eta)


def black_scholes(S, K, t, sigma, r=0.0125, put=False):
    """
    call price from Black-Scholes model
    source: https://raymond-python.blogspot.com/2018/09/pythonblack-scholes1.html
    """
    R = np.exp(r * t)  # risk-free return
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * t) / (sigma * t ** .5)
    d2 = d1 - sigma * t ** .5
    call = S * scipy.stats.norm.cdf(d1) - K / R * scipy.stats.norm.cdf(d2)
    if put:
        # call - put = S - K/R
        put = call - S + K/R
        return put
    return call
