import os


class PathEnv:

    def __call__(self, *args):
        """join path under ``self.base_path``"""
        return os.path.join(self.base_path, *args)

    def set_root(self):
        if os.name == "nt":
            self.base_path = "C:\\"
        else:
            self.base_path = "/"
        return self

    def set_path(self, base_path_default='./runtime/default/'):
        """
        set ``base_path`` if environ var BASE_PATH is missing
        also it generates that dir
        """
        self.base_path = os.getenv('BASE_PATH', base_path_default)
        ensure_dir(self.base_path)
        return self


def ensure_dir(fpath):
    """makedirs if corresponding dir is missing"""
    fdir = os.path.dirname(fpath)
    if len(fdir) > 0:
        os.makedirs(fdir, exist_ok=True)
    return fpath


def utouch(file):
    """makedirs + utime + create file"""
    ensure_dir(file)
    try:
        os.utime(file, None)
    except OSError:
        open(file, 'a').close()


def ffread(file):
    """Fast File Read"""
    with open(file, "r") as f:
        return f.read()


def ffwrite(file, text):
    """Fast File Write"""
    with open(file, "w") as f:
        return f.write(text)
