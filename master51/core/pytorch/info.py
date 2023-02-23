import numpy as np
def model_size(module, trainable_only=False):
    r = 0
    for param in module.parameters():
        if trainable_only:
            if not hasattr(param, 'requires_grad'):
                continue
            if not param.requires_grad:
                continue
        r += np.prod(param.size())
    return int(r)
