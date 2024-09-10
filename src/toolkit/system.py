from contextlib import contextmanager
import os
import socket
from datetime import datetime
import random

import numpy as np
import torch


TS_FORMAT = '%y%m%d_%H%M'
TS_LEN = 11

def set_seeds(seed=42):
    """
    Set seeds for reproducibility in PyTorch and NumPy.

    This function sets seeds for random number generators in PyTorch and NumPy to ensure
    reproducibility of results.

    Parameters:
        seed (int): Seed value for random number generation. Default is 42.

    Example:
        >>> set_seeds(seed=42)
    """
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    print(f'Seed was set to {seed}')


def system_startup(force_cpu=False):
    """
    Log system information and return the selected device.

    This function logs information about the system, including available CPUs and GPUs.
    It also allows for GPU profiling if specified.

    Parameters:
        profiling (bool, optional): Enable GPU profiling. Default is False (very expensive memorywise).
        force_cpu (bool, optional): Force the program to run on the CPU. Default is False.
        assert_gpu (bool, optional): Assert that a GPU is available. Default is True. (If CPU is forced, this is ignored.)

    Returns:
        torch.device: Selected device.

    Example:
        >>> device = system_startup(profiling=False)

    """
    device = torch.device('cuda:0') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
    print('Currently evaluating -------------------------------:')
    print(timestamp().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    if torch.cuda.is_available() and not force_cpu:
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    print(f'Running on {device}')
    print('----------------------------------------------------')
    return device

#TIME FUNCTIONS
def timestamp():
    """
    Get the current timestamp.

    Returns:
        datetime.datetime: Current timestamp.

    Example:
        >>> current_time = timestamp()

    """
    return datetime.now()

def ts_to_str(ts):
    """
    Convert a timestamp to a string.

    Parameters:
        ts (datetime.datetime): Timestamp to convert.

    Returns:
        str: Converted timestamp as a string.

    Example:
        >>> ts_str = ts_to_str(timestamp())

    """
    return ts.strftime(format=TS_FORMAT)

def str_to_ts(ts_str):
    """
    Convert a string to a timestamp.

    Parameters:
        ts_str (str): Timestamp as a string.

    Returns:
        datetime.datetime: Converted timestamp.

    Example:
        >>> ts = str_to_ts('2023-09-12 15:30:00')

    """
    return datetime.strptime(ts_str, TS_FORMAT)

@contextmanager
def set_default_tensor_type(tensor_type):
    """
    Temporarily sets the default tensor type for PyTorch operations.

    Args:
        tensor_type (torch.dtype): The new default tensor type.

    Yields:
        None

    Example:
        >>> with set_default_tensor_type(torch.cuda.FloatTensor):
        ...     # Code block where the default tensor type is set to CUDA tensors
        ...     pass
    """
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor
        
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)
