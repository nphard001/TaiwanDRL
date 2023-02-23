"""
IO stuff, file system as a database in this project
"""
import warnings
from os.path import exists
from master51 import wht
from master51 import p5

ENV_M51 = wht.PathEnv().set_root()
ENV_M51.base_path = ENV_M51('dat', 'master51')
ENV_CACHE_M51 = wht.PathEnv().set_root()
ENV_CACHE_M51.base_path = ENV_CACHE_M51('dat', 'master51', 'cache')

class PickleIOManager:

    def __init__(self, env: wht.PathEnv):
        self.env = env

    def is_file(self, *args):
        pth = self.env(*args)
        return exists(pth)

    def hit(self, *args):
        return self.is_file(*args)

    def get(self, *args):
        if not self.hit(*args):
            return None
        return p5.load(self.env(*args))

    def set(self, obj, *args):
        pth = self.env(*args)
        if self.hit(*args):
            warnings.warn(f"override cache target {pth}")
        p5.dump(obj, pth)


FILE_M51 = PickleIOManager(ENV_M51)
CACHE_M51 = PickleIOManager(ENV_CACHE_M51)
