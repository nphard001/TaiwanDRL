import numpy as np
import pandas as pd


class Tracker:
    """DataFrame manager that add new row easily, mostly for tracking training error"""

    def __init__(self, *args):
        """Define ``columns`` in DataFrame.
        Example:
            ``Tracker('idx', 'a', 'b')``
        """
        self.col = args
        self.df = pd.DataFrame([], None, args)

    def __getstate__(self):
        return [self.col, self.df]

    def __setstate__(self, state):
        self.col, self.df = state

    def add(self, *args, **kwargs):
        """Add new converted row and fills ``np.nan`` if no corresponded value"""
        d_row = {k: np.nan for k in self.col}
        for i, arg in enumerate(args):
            d_row[self.col[i]] = self.convert(arg)
        for k, v in kwargs.items():
            if k in self.col:  # input not in col will be ignored
                d_row[k] = self.convert(v)
        self.df.loc[len(self.df)] = d_row

        # prevent floating point index (tedious)
        type_dict = {k: type(v) for k, v in d_row.items()}
        self.df = self.df.astype(type_dict)
        return self

    @property
    def convert(self):
        """Define how each data converted, by default call numpy.item()"""
        if not hasattr(self, '_convert'):
            self._convert = self.get_default_convert()
        return self._convert

    def get_default_convert(self):
        def _func(x):
            if isinstance(x, np.ndarray):
                return x.item()
            return x
        return _func


class Packer:
    """Pack multiple mini-batches data and unpack it into np.ndarray"""

    def __init__(self, default_type=np.object):
        self.batches = []
        self.default_type = default_type

    def pack(self, *args):
        comps = []
        for arg in args:
            if not isinstance(arg, np.ndarray):
                arg = np.array(arg, self.default_type)
            if len(arg.shape) == 0:
                # to avoid ``ValueError('zero-dimensional arrays cannot be concatenated')``
                arg = arg.reshape(-1)
            comps.append(arg)
        self.batches.append(comps)

    def unpack(self):
        n, m = len(self.batches), len(self.batches[0])
        comps = []
        for j in range(m):
            comps.append(np.concatenate(
                [self.batches[i][j] for i in range(n)], axis=0))
        return comps
