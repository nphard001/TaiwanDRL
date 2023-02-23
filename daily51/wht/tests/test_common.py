import pytest
from daily51.wht import *


class ExpModel(BaseModel):

    def __init__(self, **params):
        if len(params) > 0:
            self.set_param(params)

    def val(self, x):
        return x**self("n", 1)


@pytest.mark.parametrize("x, n, ans", [(1.5, 2, 2.25), (2, 3, 8)])
def test_model(x, n, ans):
    model = ExpModel(n=n)
    assert model.val(x)==ans


def test_model_default():
    model = ExpModel()
    assert model.val(3.14) == 3.14
    model = ExpModel()
    model.set_param({"n": 2})
    assert model.val(2) == 4


def test_int_split():
    num = int_split(3, [1, 1])
    assert repr(list(num)) == repr(list([2, 1]))
    num = int_split(6, [100, 1])
    assert repr(list(num)) == repr(list([5, 1]))
    num = int_split(6, [100, 1, 100], min_num=2)
    assert repr(list(num)) == repr(list([2, 2, 2]))


def test_p4_identical():
    import numpy as np
    obj = np.arange(100).reshape(10, 10)
    s0 = p4.dumps_b64(obj)
    obj2 = p4.loads_b64(s0)
    assert repr(obj) == repr(obj2)


def test_path_env():
    env = PathEnv()
    env.base_path = "/testing_prefix"
    chain = env("dir1", "dir2", "dir3")
    assert "dir1" in chain
    assert "dir2" in chain
    assert "dir3" in chain


def test_tracker():
    track = Tracker("a", "b", "c")
    track.add(np.nan, np.nan, np.nan)
    track.add(3.0, 2.0, 1.0)
    track.add(b=9.0)
    track.add(1.0, c=7.0)
    assert track.df.iloc[1, 0] == 3
    assert track.df.iloc[2, 1] == 9
    assert track.df.iloc[3, 2] == 7
    print(track.df)


def test_tracker_mismatched_type():
    track = Tracker("x", "y")
    track.add()  # x is float
    track.add(y=0.5)
    with pytest.raises(Exception) as e_info:
        track.add(x=1)  # integer literal can not fit
    print(e_info)


def test_packer():
    packer = Packer()
    for i in range(3):
        a = np.arange((i+1) * 32).reshape(-1, 32)
        b = np.arange((i+1))
        print(f"--- i={i} ---")
        print(a)
        print(b)
        packer.pack(a, b)
    a, b = packer.unpack()
    print("### final a")
    print(a)
    print("### final b")
    print(b)
    assert len(b) == 6
    assert np.sum(a[:, 0]) % 32 == 0
