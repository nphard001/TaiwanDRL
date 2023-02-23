import uuid
import hashlib
import numpy as np
import tqdm


def get_logger():
    try:
        import loguru
        return loguru.logger
    except (ImportError, ModuleNotFoundError):
        import logging
        logger = logging.getLogger()
        return logger


def get_tqdm(*args, **kwargs) -> tqdm.std.tqdm:
    import tqdm.auto
    tqdm.tqdm.get_lock().locks = []
    return tqdm.auto.tqdm(*args, **kwargs)


def md5_hex(x):
    return hashlib.md5(x.encode('utf-8')).hexdigest()


def get_uuid(prefix=None, max_len=None):
    raw_uuid = uuid.uuid4().hex
    prefix = "" if prefix is None else prefix
    return prefix+raw_uuid[len(prefix):max_len]


def get_uuid_seed():
    return uuid.uuid4().int % 2147483647


def soft_update(target_dict, input_dict):
    """update target_dict without new redundant keys from input_dict"""
    for k in target_dict.keys():
        if k in input_dict:
            target_dict[k] = input_dict[k]
    return target_dict


def int_split(N, ratio_raw, min_num=None):
    """split N into M integers wrt given ratio"""
    M = len(ratio_raw)
    if min_num:
        N2 = N - min_num*M
        assert N2>=0, "N not enough"
        return int_split(N2, ratio_raw) + min_num
    ratio = np.array(ratio_raw, np.float64)
    ratio = ratio / np.sum(ratio)
    num = np.zeros([len(ratio)], dtype=np.int64)
    num += np.int64(np.floor(ratio*N))
    res = N-np.sum(num)
    quo, rem = res//M, res%M
    num += quo
    if rem>0:
        for r_id in np.argsort(ratio):
            num[r_id] += 1
            rem -= 1
            if rem==0:
                break
    return num
