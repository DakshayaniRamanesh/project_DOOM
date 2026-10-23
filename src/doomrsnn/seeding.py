import random
import numpy as np
import torch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_device(value='auto'):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but PyTorch does not see a CUDA device')
    return device
