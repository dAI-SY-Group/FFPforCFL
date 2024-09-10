import os

import numpy as np
import pandas as pd
import torch

from src.toolkit.config import yaml_to_munch
from src.data.datasets.Dataset import Dataset

def get_available_datasets():
    """
    Load 'data/configs/available_datasets.yaml' and return the dict that holds name: config path

    Returns:
        dict: name: config path
    """
    # get full path to available datasets file (lies in configs/available_datasets.yaml relative to this file) 
    available_datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'datasets', 'available_datasets.yaml')

    # load yaml file
    available_datasets = yaml_to_munch(available_datasets_path)
    return available_datasets

def update_data_config(base_config, config_file):
    """
    Update data configuration in the base configuration.

    This function loads a YAML configuration file specific to datasets, extracts relevant
    information, and updates the base configuration with the new data settings.

    Args:
        base_config (DefaultMunch): 
            The base configuration object.
        config_file (str): 
            The name of the dataset configuration file.

    Returns:
        None

    Example:
        update_data_config(config, 'classification/CIFAR10.yaml')

    Note:
        - This function requires the `yaml_to_munch` function.

    """
    config = yaml_to_munch(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'datasets', config_file))
    base_config.data = config.data
    base_config.num_classes = config.num_classes
    
def update_data_config_dataset(base_config, dataset):
    """
    Update data configuration in the base configuration.

    This function loads a YAML configuration file specific to datasets, extracts relevant
    information, and updates the base configuration with the new data settings.

    Args:
        base_config (DefaultMunch): 
            The base configuration object.
        dataset (str): 
            The name of the dataset.

    Returns:
        None

    Example:
        update_data_config_dataset(config, 'CIFAR10')

    Note:
        - This function requires the `update_data_config` function.

    """
    available_datasets = get_available_datasets()
    assert dataset in available_datasets, f'Dataset {dataset} is not available. Try one of {list(available_datasets.keys())} (See available_datasets.yaml).'
    update_data_config(base_config, available_datasets[dataset])

def get_fed_distribution_identifier(client_list, distribution_config, seed):
    return f'NC_{len(client_list)}_' + '_'.join([f'{k}_{v}' for k, v in distribution_config.items() if 'path' not in k and 'fingerprint' not in k]) + f'_seed_{seed}'

class DataLoaderIterator:
    def __init__(self, data_loaders):
        """
        Iterator over multiple data loaders.

        Args:
            data_loaders (list): List of data loaders.

        Returns:
            tuple: Batch data and labels.

        __len__:
            Returns the total number of batches across all data loaders.

        __call__:
            Yields batches of data and labels from each data loader.
        """
        self.data_loaders = data_loaders
    
    def __call__(self):
        for i, data_loader in enumerate(self.data_loaders):
            for x, y in data_loader:
                yield x, y

    def __len__(self):
        return sum([len(dl) for dl in self.data_loaders])

def split_dataset(dataset, split=(0.9, 0.1), shuffle=True, seed=42, ds1_transforms=None, ds2_transforms=None, ds1_target_transforms=None, ds2_target_transforms=None, dataset_class=Dataset, *args, **kwargs):
    """
    Split a dataset into two datasets.

    Args:
        dataset (Dataset): The original dataset.
        split (tuple): A tuple specifying the split ratio.
        shuffle (bool): Whether to shuffle the dataset before splitting.
        seed (int): Seed for randomization.
        ds1_transforms (list, 'same', optional): Transforms for the first dataset.
        ds2_transforms (list, 'same', optional): Transforms for the second dataset.
        ds1_target_transforms (list, 'same', optional): Target transforms for the first dataset.
        ds2_target_transforms (list, 'same', optional): Target transforms for the second dataset.
        dataset_class (type, optional): Class type of the dataset.

    Returns:
        Dataset: The first split of the dataset.
        Dataset: The second split of the dataset.
    """
    assert sum(split) == 1, 'split must sum to 1'
    if shuffle:
        torch.manual_seed(seed)
        indices = torch.randperm(len(dataset)).tolist()
    else:
        indices = list(range(len(dataset)))
    split1 = int(split[0] * len(dataset))
    if ds1_transforms == 'same':
        ds1_transforms = dataset.transforms
    if ds2_transforms == 'same':
        ds2_transforms = dataset.transforms
    if ds1_target_transforms == 'same':
        ds1_target_transforms = dataset.target_transforms
    if ds2_target_transforms == 'same':
        ds2_target_transforms = dataset.target_transforms
    ds1 = dataset_class(data = dataset.data[indices[:split1]], targets = dataset.targets[indices[:split1]], transforms=ds1_transforms, target_transforms=ds1_target_transforms, *args, **kwargs)
    ds2 = dataset_class(data = dataset.data[indices[split1:]], targets = dataset.targets[indices[split1:]], transforms=ds2_transforms, target_transforms=ds2_target_transforms, *args, **kwargs)
    return ds1, ds2

# get the distribution of classes in a dataset given the list of labels and the number of classes
def get_class_distribution(labels, num_classes):
    """
    Get the distribution of classes in a dataset.

    Args:
        labels (list): List of class labels.
        num_classes (int): Total number of classes.

    Returns:
        np.array: Array containing class counts.
    """
    counts = np.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    return counts

def get_unique_targets(dataloader):
    """
    Get unique targets from a data loader.

    Args:
        dataloader (DataLoader): PyTorch data loader.

    Returns:
        np.array: Array containing unique targets.
    """
    return np.unique(dataloader.dataset.targets)

class Normalizer(object):
    """
    Normalize input data.

    Args:
        mean (float, optional): Mean value. Default is None.
        std (float, optional): Standard deviation. Default is None.
        mode (str, optional): Normalization mode ('batch', 'stat', 'single'). Default is 'batch'.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    def __init__(self, mean=None, std=None, mode='batch'):
        self.mean = mean
        self.std = std
        self.mode = mode
        if mode == 'stat':
            assert mean is not None and std is not None, 'If data is to be normalized based on specific statistics, you need to provide mean and std.mean'
            self.norm_fn = self.stat_norm
        elif mode == 'batch':
            self.norm_fn = self.scale_01
        elif mode == 'single':
            self.norm_fn = self.scale_single_01
        else: 
            raise ValueError(f'The mode {mode} is not implemented yet.')
    
    def __call__(self, tensor):
        return self.norm_fn(tensor)

    def stat_norm(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.div_(s).sub_(m)
        return tensor
    
    def scale_01(self, tensor):
        return 1 - (tensor.max()-tensor) / ((tensor.max() - tensor.min())+1e-5)
   
    def scale_single_01(self, tensor):
        t = []
        for x in tensor:
            t.append(self.scale_01(x))
        return torch.stack(t)
 
class DeNormalizer(object):
    """
    De-normalize input data.

    Args:
        mean (float): Mean value.
        std (float): Standard deviation.

    Returns:
        torch.Tensor: De-normalized tensor.

    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_statistics(dataset):
    """
    Get mean and standard deviation of a dataset.

    Args:
        dataset (Dataset): The dataset.

    Returns:
        list: List containing mean values.
        list: List containing standard deviation values.
    """
    cc = torch.cat([dataset[i][0].reshape(3, -1) for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std

def stat_to_tensor(stat):
    """
    Convert statistics to tensor format.

    Args:
        stat (list): List of statistics.

    Returns:
        torch.Tensor: Converted tensor.

    """
    return torch.as_tensor(stat)[:, None, None].float()

def get_default_dataset_shape(dataset):
    """
    Get default dataset shape.

    Args:
        dataset (str): Name of the dataset.

    Returns:
        tuple: Default dataset shape.

    """
    available_datasets = get_available_datasets()
    assert dataset in available_datasets, f'Dataset {dataset} is not available. Try one of {list(available_datasets.keys())} (See available_datasets.yaml).'
    config = yaml_to_munch(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'datasets', available_datasets[dataset]))
    return config.data.shape

def get_dataset_mean_dev_tensors(dataset):
    """
    Get mean and standard deviation tensors for a dataset.

    Args:
        dataset (str): Name of the dataset.

    Returns:
        torch.Tensor: Mean tensor.
        torch.Tensor: Standard deviation tensor.

    """
    available_datasets = get_available_datasets()
    assert dataset in available_datasets, f'Dataset {dataset} is not available. Try one of {list(available_datasets.keys())} (See available_datasets.yaml).'
    config = yaml_to_munch(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'datasets', available_datasets[dataset]))

    dm = stat_to_tensor(config.data.mean)
    ds = stat_to_tensor(config.data.std)
    return dm, ds

def get_dataset_mean_dev_series(dataset, tensor=None):
    """
    Get mean and standard deviation series for a dataset.

    Args:
        dataset (str): Name of the dataset.
        tensor (list, optional): List of series columns. Default is None.

    Returns:
        pd.Series: Mean series.
        pd.Series: Standard deviation series.

    """
    available_datasets = get_available_datasets()
    assert dataset in available_datasets, f'Dataset {dataset} is not available. Try one of {list(available_datasets.keys())} (See available_datasets.yaml).'
    config = yaml_to_munch(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'datasets', available_datasets[dataset]))
    #if you want a tensor you need to specify the series columns as a list
    if tensor:
        dmd = dict(pd.Series(config.data.mean))
        dsd = dict(pd.Series(config.data.std))
        dm = []
        ds = []
        for name in tensor:
            dm.append(dmd[name])
            ds.append(dsd[name])
        dm = stat_to_tensor(config.data.mean)
        ds = stat_to_tensor(config.data.std)
    else:
        dm = pd.Series(config.data.mean)
        ds = pd.Series(config.data.std)
    return dm, ds

