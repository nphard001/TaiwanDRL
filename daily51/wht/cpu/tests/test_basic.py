import pytest
import numpy as np
from numpy.testing import assert_array_equal
import daily51.wht.cpu as qcpu


def test_score():
    a = np.arange(6).reshape(3, 2)
    b = [[0, 0], [0.5, 0.5], [1, 1]]
    assert_array_equal(qcpu.order_score(a, axis=0), b)
    b = [[0, 1], [0, 1], [0, 1]]
    assert_array_equal(qcpu.order_score(a, axis=1), b)
    a = [[-1, np.nan], [0, 2]]
    b = qcpu.order_score(a, axis=None)
    assert b[0, 1] == 0
    assert b[1, 1] == 1

    a = [[0, 1], [2, 3]]
    b = qcpu.z_score(a)
    assert b[0, 0] < b[1, 1]

    a = [[1, 1, 1], [0, 0, 0]]
    b = qcpu.z_score(a, axis=None)
    assert b[0, 0] > b[1, 1]
    b = qcpu.z_score(a, axis=1)
    assert b[0, 0] == b[1, 1]


def test_scalar_fn():
    a, b, c = qcpu.sigmoid([-100, 0, 100])
    assert a < 0.1
    assert b == 0.5
    assert c > 0.9
    a = np.linspace(-1, 1, 11)
    b = qcpu.relu(a)
    assert np.sum(b>=0) == 11
    assert np.sum(b>0) == 5


def test_cartesian_product():
    a = qcpu.cartesian_product_itertools([[1, 2], [3, 4]])
    b = [[1, 3], [1, 4], [2, 3], [2, 4]]
    assert_array_equal(a, b)
    a = qcpu.cartesian_product_itertools(np.arange(6).reshape(2, 3))
    assert len(a) == 9
    a = qcpu.cartesian_product_itertools(np.arange(6).reshape(3, 2))
    assert len(a) == 8