import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_worker_init_fn(seed):
    def worker_init_fn(worker_id):  # set for each dataloader worker
        set_all_seeds((seed+1)*(worker_id+1))
    return worker_init_fn


def set_deterministic(seed):
    set_all_seeds(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def ensure_pytorch(pytorch_or_numpy):
    """Ensure output is a pytorch object.
    ndarray: as_tensor
    tensor, model, optimizer, ...: remains the same
    """
    if isinstance(pytorch_or_numpy, (np.ndarray)):
        pytorch_tns = torch.as_tensor(pytorch_or_numpy)
    else:
        pytorch_tns = pytorch_or_numpy
    return pytorch_tns


# TODO test CPU mode
def make_device(device_name='cuda', non_blocking=True):
    """
    Detect device_count and make (GPU, CPU) function pairs:
        1. GPU(pytorch_or_numpy): Ensure a tensor in the device you want
        2. CPU(pytorch_tensor): Copy back to memory and convert into np.ndarray
        3. device name
    Usage:
        GPU, CPU, device = make_device()
    Notes:
        If something can be done with pytorch, you won't convert them into numpy.
        So ``.numpy()`` is always called
    """
    cpu_only = False
    if device_name == 'cpu':
        cpu_only = True
    if torch.cuda.device_count() == 0:
        # TODO downgrade warning
        cpu_only = True

    if not cpu_only:
        def _GPU_single(pytorch_or_numpy):
            return ensure_pytorch(pytorch_or_numpy).to(device_name, non_blocking=non_blocking)

        def _CPU_single(pytorch_tensor):
            return pytorch_tensor.cpu().data.numpy()
    else:
        device_name = 'cpu'

        def _GPU_single(pytorch_or_numpy):
            return ensure_pytorch(pytorch_or_numpy)

        def _CPU_single(pytorch_tensor):
            return pytorch_tensor.data.numpy()

    def _many_to_many(fn_single):
        def fn_many(*args):
            if len(args) == 1:
                return fn_single(args[0])
            return [fn_single(arg) for arg in args]
        return fn_many
    return _many_to_many(_GPU_single), _many_to_many(_CPU_single), device_name

