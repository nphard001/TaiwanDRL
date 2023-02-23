import itertools
import numpy as np


def cartesian_product_itertools(arrays):
    """source: https://stackoverflow.com/a/45378609"""
    return np.array(list(itertools.product(*arrays)))


def sigmoid(arr_in):
    x = np.array(arr_in)
    return 1 / (1 + np.exp(-x))


def relu(arr_in):
    x = np.array(arr_in)
    return np.where(x > 0, x, 0)
