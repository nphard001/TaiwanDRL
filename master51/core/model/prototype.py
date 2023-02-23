from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from master51 import wht


class PytorchStyleSavable:

    def state_dict(self) -> dict:
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict


class Model(ABC, PytorchStyleSavable):

    @property
    def param(self):
        if not hasattr(self, "_param"):
            self._param = self.get_param_default()
        return self._param

    def __call__(self, key: str):
        """Syntax sugar fetching param"""
        return self.param[key]

    def set_param(self, input: dict):
        for key in input:
            self.param[key] = input[key]
        return self

    @abstractmethod
    def get_param_default(self) -> dict:
        """Build a new default hyperparameter dict"""

    @property
    def md5(self) -> str:
        """A quick hash index of param"""
        return wht.md5_hex(str(self.param))


class Preprocessor(ABC):

    @abstractmethod
    def set_fold(self, name: str):
        """Change preset fold definition"""


class Reporter(ABC, PytorchStyleSavable):

    @property
    def rows(self) -> list:
        if not hasattr(self, "_rows"):
            self._rows = []
        return self._rows

    def append(self, row):
        self.rows.append(row)

    @abstractmethod
    def report(self, *args, **kwargs):
        """record and show training details"""

    def state_dict(self) -> dict:
        return {"rows": self.rows}  # Only rows are matter in a reporter

    def load_state_dict(self, state_dict):
        self._rows = state_dict["rows"]


class ModelGenerator(ABC):

    @abstractmethod
    def collect(self, *args, **kwargs):
        """Generate model and collect results"""
