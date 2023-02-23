"""
Statistics
"""
import numpy as np

################################################################
# TODO: TEST
################################################################


def _order_score_1d(arr_in):
    arr_flat = arr_in.copy().reshape(-1)

    # nan handling: https://stackoverflow.com/a/41215209
    for i in range(len(arr_flat)):
        if arr_flat[i] != arr_flat[i]:
            arr_flat[i] = -np.inf

    arr_out = np.zeros(arr_flat.shape, dtype=np.float32)
    keys = np.unique(arr_flat)
    factor = max(1, len(keys)-1)
    mapping = {}
    for i, key in enumerate(keys):
        mapping[key] = i/factor
    for i in range(len(arr_out)):
        arr_out[i] = mapping[arr_flat[i]]
    return arr_out.reshape(arr_in.shape)


def order_score(arr_in, axis=0):
    """sort base on ``np.unique`` and map the order score to [0, 1]"""
    arr = np.array(arr_in)
    if axis is None:
        return _order_score_1d(arr)
    return np.apply_along_axis(_order_score_1d, axis, arr)


def z_score(arr_in, axis=None, eps=1e-9):
    """normalize ``arr_in`` into z-score"""

    def _z_score_1d(arr_in):
        sd = arr_in.std()
        return (arr_in - arr_in.mean()) / (sd + eps)

    arr = np.array(arr_in)
    if axis is None:
        return _z_score_1d(arr)
    return np.apply_along_axis(_z_score_1d, axis, arr)


def qtable(arr_in, n):
    """view cut points / quantiles quickly"""
    arr = np.array(arr_in)
    if len(arr.shape)==1:
        arr = arr.reshape(-1, 1)
    p = np.linspace(0, 1, n)
    qs = [np.quantile(arr[:, icol], p) for icol in range(arr.shape[1])]
    return np.stack([p, *qs], axis=1)


def get_ecdf_fn(input):
    """
    Empirical CDF from ``statsmodels`` package
    seperate that function so only import ``statsmodels`` when needed
    """
    import warnings
    # NOTE: FutureWarning: pandas.util.testing is deprecated.
    #       Use the functions in the public API at pandas.testing instead.
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
            message='.*testing is deprecated.*')
        from statsmodels.distributions.empirical_distribution import ECDF
    return ECDF(input)

