from collections import defaultdict

import numpy as np
import pandas as pd
import torch

class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for handling input data and labels.

    Args:
        data (array-like): Input data.
        targets (array-like): Labels corresponding to the input data.
        transforms (callable, optional): A function/transform to apply to the input data.
        target_transforms (callable, optional): A function/transform to apply to the labels.
        manual_transform (bool, optional): Whether to manually convert data and targets to tensors.
        num_classes (int, optional): Number of class labels in the dataset.

    Attributes:
        data (torch.Tensor or array-like): Input data.
        targets (torch.Tensor or array-like): Labels corresponding to the input data.
        transforms (callable, optional): A function/transform to apply to the input data.
        target_transforms (callable, optional): A function/transform to apply to the labels.
        target_distribution (dict): Distribution of class labels in the dataset.
        num_classes (int): Number of class labels in the dataset.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the item at the specified index.
        __str__(): Returns a string representation of the dataset.
        __repr__(): Returns a detailed string representation of the dataset.

    """
    def __init__(self, data, targets, transforms=None, target_transforms=None, manual_transform=False, num_classes=None):
        """
        Initialize Dataset with input data, labels, and optional transformations.

        If `manual_transform` is True, data and targets are converted to tensors manually.

        Args:
            data (array-like): Input data.
            targets (array-like): Labels corresponding to the input data.
            transforms (callable, optional): A function/transform to apply to the input data.
            target_transforms (callable, optional): A function/transform to apply to the labels.
            manual_transform (bool, optional): Whether to manually convert data and targets to tensors.

        """
        if manual_transform:
            self.data = torch.FloatTensor(np.array(data))
            self.targets = torch.LongTensor(targets)
        else:
            self.data = np.array(data)
            self.targets = np.array(targets, dtype=np.int64)
        self.transforms = transforms
        self.target_transforms = target_transforms
        #calculate and store number of class labels of the dataset
        self.target_distribution = defaultdict(int, dict(pd.Series(self.targets).value_counts().sort_index()))
        self.num_classes = len(self.target_distribution) if num_classes is None else num_classes

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.

        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index. If transforms are specified, they are applied to the data and labels.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the input data and its corresponding label.

        """
        x = self.data[idx]
        y = self.targets[idx]
        if self.transforms:
            x = self.transforms(x)
        if self.target_transforms:
            y = self.target_transforms(y)
        return x, y
    
    def __str__(self):
        """
        Returns a string representation of the dataset.

        Returns:
            str: String representation of the dataset.

        """
        return f'Dataset size: {len(self)}. Input shape: {self.data.shape}. Output shape: {self.targets.shape}.\nTransforms: {self.transforms}.\nTarget Transforms: {self.target_transforms}.\nTarget Distribution: {self.target_distribution}.'
    
    def __repr__(self):
        """
        Returns a detailed string representation of the dataset.

        Returns:
            str: Detailed string representation of the dataset.

        """
        return self.__str__()