import os

import numpy as np
import torchvision
from torch.utils.data import DataLoader

from src.data.utils import split_dataset

from src.data.datasets.Dataset import Dataset


def _build_cifar10(datapath, train_transformations, val_transformations):
    """
    Builds CIFAR-10 datasets for training and testing.

    Args:
        datapath (str): Path to store the CIFAR-10 dataset.
        train_transformations (torchvision.transforms.Compose, optional): Transformations for training set. Default is transforms.ToTensor().
        val_transformations (torchvision.transforms.Compose, optional): Transformations for validation set. Default is transforms.ToTensor().

    Returns:
        Dataset: Training set.
        Dataset: Testing set.

    """
    trn_set = torchvision.datasets.CIFAR10(root=datapath, train=True, download=True)
    tst_set = torchvision.datasets.CIFAR10(root=datapath, train=False, download=True)
    trn_set = Dataset(trn_set.data, trn_set.targets, train_transformations)
    tst_set = Dataset(tst_set.data, tst_set.targets, val_transformations)
    return trn_set, tst_set


def _build_med_mnist(datapath, train_transformations, val_transformations):
    """
    Builds a MedMNIST dataset for training, testing, and validation.

    Args:
        datapath (str): Path to the MedMNIST dataset.
        train_transformations (torchvision.transforms.Compose, optional): Transformations for training set. Default is transforms.ToTensor().
        val_transformations (torchvision.transforms.Compose, optional): Transformations for validation and test sets. Default is transforms.ToTensor().

    Returns:
        Dataset: Training set.
        Dataset: Testing set.
        Dataset: Validation set.

    """
    class MedMNIST(Dataset):
        """
        MedMNIST dataset class. Dataset from https://medmnist.com/

        Args:
            datapath (str): Path to the MedMNIST dataset.
            split (str): Specifies the dataset split (e.g., 'train', 'val', 'test').
            transform (callable, optional): A function/transform to apply to the data. Default is transforms.ToTensor().

        Attributes:
            datapath (str): Path to the MedMNIST dataset.
            split (str): Specifies the dataset split (e.g., 'train', 'val', 'test').
            transforms (callable, optional): A function/transform to apply to the data.
            data (numpy.ndarray): Array of images.
            targets (numpy.ndarray): Array of labels.

        Methods:
            get_np(index): Returns the numpy representation of an image at the specified index.
            __getitem__(index): Gets an item (image and label) from the dataset at the specified index.
            __len__(): Returns the total number of samples in the dataset.

        """
        def __init__(self, datapath, split, transforms):
            self.datapath = datapath
            self.split = split
            npz_data = np.load(datapath)
            data = npz_data[f'{split}_images']
            if 'chestmnist' in datapath:
                targets = npz_data[f'{split}_labels'][:,6] # ChestMNIST has 14 labels, but we only want one of them to get single label binary classification task (in this case pneumonia!)
            else:
                targets = npz_data[f'{split}_labels']
            targets = targets.squeeze()
            super().__init__(data=data, targets=targets, transforms=transforms, target_transforms=None, manual_transform=False)

            assert len(self.data) == len(self.targets)

        def get_np(self, index):
            """
            Returns the numpy representation of an image at the specified index.

            Args:
                index (int): Index of the image.

            Returns:
                numpy.ndarray: Numpy array representing the image.

            """
            return self.data[index]

        def __getitem__(self, index):
            """
            Gets an item (image and label) from the dataset at the specified index.

            Args:
                index (int): Index of the item.

            Returns:
                tuple: Tuple containing the transformed image and its label.

            """
            sample = self.data[index]
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample, self.targets[index]

        def __len__(self):
            return len(self.targets)

    return MedMNIST(datapath, 'train', train_transformations), MedMNIST(datapath, 'test', val_transformations), MedMNIST(datapath, 'val', val_transformations)



def get_central_dataloader(config, trn_transformations, val_transformations):
    """
    Returns central data loaders for training, testing, and validation sets.

    This function prepares data loaders for the specified dataset using the provided configurations
    and transformations.

    Parameters:
        config (Munch): Configuration object containing dataset and training settings.
        trn_transformations (callable): Transformation function for training set.
        val_transformations (callable): Transformation function for validation set.

    Returns:
        tuple: A tuple containing three data loaders (training, testing, and validation).
               If a specific set is not available, the corresponding loader will be None.

    Raises:
        (various): This function may raise various exceptions depending on the specific dataset
                   loading and transformation functions used.

    Example:
        >>> config = build_config('...')
        >>> trn_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        >>> val_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        >>> trn_loader, tst_loader, val_loader = get_central_dataloader(config, trn_transforms, val_transforms)

    Note:
        If the validation set is not provided but `create_validation_split` is enabled in the
        configuration, a split will be created from the training set.
    """
    get_dataset = eval(f'get_{config.data.dataloader}_dataset')
    trn_set, tst_set, val_set = get_dataset(config, trn_transformations, val_transformations)

    if val_set is None and config.data.create_validation_split:
        print(f'Creating validation split of {config.data.create_validation_split} from training set')
        trn_set, val_set = split_dataset(trn_set, (1-config.data.create_validation_split, config.data.create_validation_split), shuffle = True, ds1_transforms=trn_transformations, ds2_transforms=val_transformations)

    if trn_set is not None:
        trn_set.num_classes = config.data.num_classes
        trn_loader = DataLoader(trn_set, batch_size=min(config.training.batch_size, len(trn_set)), shuffle=True, drop_last=True)
    else:
        trn_loader = None

    if tst_set is not None:
        tst_set.num_classes = config.num_classes
        tst_loader = DataLoader(tst_set, batch_size=min(config.training.batch_size, len(tst_set)), shuffle=False, drop_last=False)
    else:
        tst_loader = None

    if val_set is not None:
        val_set.num_classes = config.num_classes
        val_loader = DataLoader(val_set, batch_size=min(config.training.batch_size, len(val_set)), shuffle=False, drop_last=False)
    else:
        val_loader = None

    print(f'Created central dataloaders for {config.data.dataset} dataset. Transformed them with:')
    print(str(trn_transformations))

    print(f'Batchsize: {config.training.batch_size} | Trainbatches: {len(trn_loader) if trn_loader is not None else 0} | Testbatches: {len(tst_loader) if tst_loader is not None else 0} | Validationbatches: {len(val_loader) if val_loader is not None else 0}')
    return trn_loader, tst_loader, val_loader


def get_image_dataset(config, trn_transformations, val_transformations):
    """
    Get image dataset.

    Args:
        config: Configuration (Munch) object.
        trn_transformations: Training transformations.
        val_transformations: Validation transformations.

    Returns:
        Dataset: Training dataset.
        Dataset: Testing dataset.
        Dataset: Validation dataset.

    """
    datapath = os.path.expanduser(config.dataset_path)

    if config.data.dataset == 'CIFAR10':
        trn_set, tst_set = _build_cifar10(datapath, trn_transformations, val_transformations)
        val_set = None
    elif config.data.dataset in ['MedMNISTBlood', 'MedMNISTBreast', 'MedMNISTChest', 'MedMNISTDerma', 'MedMNISTOCT', 'MedMNISTOrganA', 'MedMNISTOrganC', 'MedMNISTOrganS', 'MedMNISTPath', 'MedMNISTPneumonia', 'MedMNISTRetina', 'MedMNISTTissue']:
        if config.data.prepped_version:
            datapath = os.path.join(datapath, 'MedMNIST', f'{config.data.dataset[8:]}', f'{config.data.dataset[8:].lower()}mnist_{config.data.prepped_version}.npz')
        else:
            datapath = os.path.join(datapath, 'MedMNIST', f'{config.data.dataset[8:]}', f'{config.data.dataset[8:].lower()}mnist.npz')
        trn_set, tst_set, val_set = _build_med_mnist(datapath, trn_transformations, val_transformations)
    else:
        raise ValueError(f'Dataloaders for the {config.data.dataset} dataset are not yet implemented!')
    return trn_set, tst_set, val_set
