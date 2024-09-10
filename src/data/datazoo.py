import numpy as np

from torch.utils.data import DataLoader

from src.data.utils import get_available_datasets, update_data_config, stat_to_tensor
from src.data.transforms import get_transforms
from src.data.datasets.Dataset import Dataset
from src.data.dataloader.central import get_central_dataloader
from src.data.dataloader.federated import get_federated_dataloader


def get_dataloaders(dataset, is_federated, config):
    """
    Get the dataloaders for the given dataset

    Args:
        dataset (str): name of the dataset
        is_federated (bool): whether to return a federated dataloader or a central dataloader
        config (Munch or str): either a Munch object containing the config for the dataset or the name of the dataset to load a standard config for

    Returns:
        tuple: either (trn_loader, tst_loader, val_loader) or (fed_data)
        
    """
    if not isinstance(dataset, str): #in this case config.data is already a Munch object and the user wanrs to use a custom dataset config
        dataset = config.data.dataset
    available_datasets = get_available_datasets()
    assert dataset in available_datasets.keys(), f'Dataset {dataset} is not available! Available datasets are {available_datasets.keys()}'

    #str or nonetype
    if isinstance(config.data, (str, type(None))): #user only defined the dataset name in config --> loading standard config for that dataset from src datazoo configs
        update_data_config(config, available_datasets[dataset])
        print(f'Using standard config for dataset {dataset}: {config.data}. Loading config from src/data/configs/datasets/')

    trn_transformations = get_transforms(config.data.train_transformations, (stat_to_tensor(config.data.mean), stat_to_tensor(config.data.std))) if config.data.train_transformations is not None else None
    val_transformations = get_transforms(config.data.val_transformations, (stat_to_tensor(config.data.mean), stat_to_tensor(config.data.std))) if config.data.val_transformations is not None else None

    if is_federated:
        dataloaders = get_federated_dataloader(config, trn_transformations, val_transformations) # returns "FederatedDataloaders object, which holds the subsets as dicts"
    else:
        dataloaders = get_central_dataloader(config, trn_transformations, val_transformations) # returns "trn_loader, tst_loader, val_loader"

    if 'RandomCrop' in config.data.train_transformations.keys():
        config.data.shape[1] = config.data.train_transformations['RandomCrop'][0][0]
        config.data.shape[2] = config.data.train_transformations['RandomCrop'][0][1]
    if 'Resize' in config.data.train_transformations.keys():
        config.data.shape[1] = config.data.train_transformations['Resize'][0][0]
        config.data.shape[2] = config.data.train_transformations['Resize'][0][1]
    print(f'Final data shape: {config.data.shape}')
    return dataloaders


def get_single_class_tst_dataloaders(config):
    #get basic central test dataloader
    _, full_test_loader, _ = get_dataloaders(config.data.dataset, False, config)
    tst_set = full_test_loader.dataset
    classwise_dataloaders = {}
    for _class in np.unique(tst_set.targets):
        indices = np.where(tst_set.targets == _class)[0]
        class_tst_set = Dataset(np.array(tst_set.data)[indices], np.array(tst_set.targets)[indices], tst_set.transforms, tst_set.target_transforms)
        class_tst_loader = DataLoader(class_tst_set, batch_size=min(config.training.batch_size, len(tst_set)), shuffle=False, drop_last=False)
        classwise_dataloaders[_class] = class_tst_loader
    return classwise_dataloaders