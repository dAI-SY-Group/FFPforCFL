import os

import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset as TorchDataset

from src.data.datasets.Dataset import Dataset

from src.data.splits import disjoint_shards_split, overlapping_shards_split

from src.data.utils import get_fed_distribution_identifier

from fedlab.utils.dataset.partition import CIFAR10Partitioner


class ClientDataset(Dataset):
    """
    Custom dataset class for a specific client in a federated learning scenario. Inherits from SimpleAI.data.datasets.bases.Dataset

    Args:
        id (int): Identifier for the client.
        data (array-like): Input data.
        targets (array-like): Labels corresponding to the input data.
        transforms (callable, optional): A function/transform to apply to the input data.
        target_transforms (callable, optional): A function/transform to apply to the labels.
        manual_transform (bool, optional): Whether to manually convert data and targets to tensors.
        num_classes (int, optional): Number of class labels in the dataset.

    Attributes:
        id (int): Identifier for the client.
        Inherits all attributes from the parent class Dataset.

    Methods:
        __init__(id, data, targets, transforms=None, target_transforms=None): Initializes ClientDataset.
        __str__(): Returns a string representation of the client dataset.

    """
    def __init__(self, id, data, targets, transforms=None, target_transforms=None, manual_transform=False, num_classes=None):
        """
        Initialize ClientDataset for a specific client.

        Args:
            id (int): Identifier for the client.
            data (array-like): Input data.
            targets (array-like): Labels corresponding to the input data.
            transforms (callable, optional): A function/transform to apply to the input data.
            target_transforms (callable, optional): A function/transform to apply to the labels.

        """
        super().__init__(data, targets, transforms, target_transforms, manual_transform, num_classes)
        self.id = id
    
    def __str__(self):
        """
        Returns a string representation of the client dataset.

        Returns:
            str: String representation of the client dataset.

        """
        return f'ClientDataset for Client {self.id}\n' + super().__str__()

class FederatedDataset(TorchDataset):
    """
    Custom dataset class for federated learning.

    Args:
        dataset (torch.utils.data.Dataset): Centralized dataset to be federated.
        client_list (list): List of client identifiers.
        distribution_config (Munch): Configuration object for data distribution.
        preset_label_distribution (dict, optional): Predefined label distribution for clients.
        transforms (callable, optional): A function/transform to apply to the input data.
        seed (int, optional): Seed value for reproducibility.

    Attributes:
        dataset (torch.utils.data.Dataset): Centralized dataset that will be split based on distribution configuration.
        targets (list): List of labels corresponding to the dataset.
        client_list (list): List of client identifiers.
        distribution_config (Munch): Configuration object for data distribution.
        preset_label_distribution (dict): Predefined label distribution for clients.
        transforms (callable): A function/transform to apply to the input data.
        seed (int): Seed value for reproducibility.
        fed_data_identifier (str): Identifier for federated dataset.
        distribution_dump_path (str): Path to store distribution information.
        idx_distribution (dict): Index distribution for clients.
        client_datasets (dict): Dictionary to store ClientDataset objects.

    Methods:
        __init__(dataset, client_list, distribution_config, preset_label_distribution=None, transforms=torchvision.transforms.ToTensor(), seed=42): Initializes FederatedDataset.
        __len__(): Returns the number of clients.
        __str__(): Returns a string representation of the dataset.
        __repr__(): Returns a string representation of the dataset.
        __getitem__(index): Retrieves a ClientDataset based on the given index.
        get_client_dataset(index, reload=False): Retrieves or reloads a ClientDataset for a specific client.
        get_index_distribution(preset_global_distribution=None): Calculates the index distribution for clients.

    """
    def __init__(self, dataset, client_list, distribution_config, preset_label_distribution=None, transforms = torchvision.transforms.ToTensor(), seed=42):
        """
        Initialize FederatedDataset.

        Args:
            dataset (torch.utils.data.Dataset): Centralized dataset to be federated.
            client_list (list): List of client identifiers.
            distribution_config (object): Configuration object for data distribution.
            preset_label_distribution (dict, optional): Predefined label distribution for clients.
            transforms (callable, optional): A function/transform to apply to the input data.
            seed (int, optional): Seed value for reproducibility.

        """
        assert distribution_config is not None, 'distribution_config must not be None for FederatedDatasets!'
        self.dataset = dataset
        self.targets = [sample[1] for sample in self.dataset]
        self.client_list = client_list
        self.distribution_config = distribution_config
        self.preset_label_distribution = preset_label_distribution
        self.transforms = transforms
        self.seed = seed
        
        self.fed_data_identifier = get_fed_distribution_identifier(self.client_list, self.distribution_config, self.seed) 
        self.distribution_dump_path = os.path.join(self.distribution_config.path, f'{self.fed_data_identifier}.tdump')

        #self.class_distribution = class_distribution
        self.idx_distribution = self.get_index_distribution(preset_label_distribution)

        self.client_datasets = {client: None for client in self.client_list}

        for client in self.client_list:
            self.get_client_dataset(client, True)

    def __len__(self):
        """
        Returns the number of clients.

        Returns:
            int: Number of clients.

        """
        return len(self.client_list)

    def __str__(self):
        return f'FederatedDataset:\n  Partition Mode: {self.distribution_config.partition_mode}\n  Clients: {self.client_list}\n  Samples per Client: {[len(ds) if ds is not None else 0 for ds in self.client_datasets.values()]}'

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, index):
        """
        Retrieves a ClientDataset based on the given index.

        Args:
            index: Index of the client.

        Returns:
            ClientDataset: ClientDataset object for the specified client.

        """
        assert index in self.client_list or index in list(range(len(self.client_list))), f'The index {index} is not part of the client_list ({self.client_list})!'
        if index not in self.client_list:
            index = self.client_list[index]
        if self.client_datasets[index] is None:
            self.get_client_dataset(index, True)
        return self.client_datasets[index]

    def get_client_dataset(self, index, reload=False):
        """
        Retrieves or reloads a ClientDataset for a specific client.

        Args:
            index: Index of the client.
            reload (bool, optional): Whether to reload the dataset.

        Returns:
            ClientDataset: ClientDataset object for the specified client.

        """
        assert index in self.client_list or index in list(range(len(self.client_list))), f'The index {index} is not part of the client_list ({self.client_list})!'
        if reload or self.client_datasets[index] is None:
            client_data = []
            client_targets = []
            for sample_index in self.idx_distribution[index]:
                client_data.append(np.array(self.dataset[sample_index][0]))
                client_targets.append(self.dataset[sample_index][1])
            if len(client_targets) == 0:
                self.client_datasets[index] = None
            else:
                self.client_datasets[index] = ClientDataset(index, client_data, client_targets, self.transforms, None, False, self.dataset.num_classes)
        return self.client_datasets[index]
        
    def get_index_distribution(self, preset_global_distribution=None):
        """
        Calculates the index distribution (i.e. how samples of the whole central base dataset are distributed between clients) for clients based on distribution configuration or preset.

        Args:
            preset_global_distribution (dict, optional): Predefined global label distribution.

        Returns:
            dict: Index distribution for clients.

        """
        if os.path.exists(self.distribution_dump_path) and not self.distribution_config.reload:
            print(f'Loading distribution from {self.distribution_dump_path}')
            return torch.load(self.distribution_dump_path)
        distribution = {}

        num_clients = len(self.client_list)
        num_classes = self.dataset.num_classes
        N = len(self.dataset)

        if preset_global_distribution is not None:
            #based on a given global preset distribution create a new index distribution for the targets that follows the same class distribution
            def create_index_distribution(targets, global_preset_distribution):
                #from the global_distribution create a matric where each row is a client and each column is a class, the value is the corresponding nnumber of samples each client has for that class
                num_clients = len(global_preset_distribution)
                num_classes = len(np.unique(targets))
                client_distribution = np.zeros((num_clients, num_classes))+1e-10
                for client, client_distribution_ in global_preset_distribution.items():
                    for class_, num_samples in client_distribution_.items():
                        client_distribution[client, class_] = num_samples
                #transform the total number of samples for each class to a distribution
                p_client_distribution = client_distribution / np.sum(client_distribution, axis=0, keepdims=True)
                #for each class get the indices of the target list
                class_indices = {class_: np.where(targets == class_)[0] for class_ in np.unique(targets)}
                class_avaliable_samples = {class_: len(class_indices[class_]) for class_ in class_indices}
                #shuffle the indices for each class
                for class_ in class_indices:
                    np.random.shuffle(class_indices[class_])

                #create a new index distribution that follows the same distribution as the global preset distribution
                target_index_distribution = {}
                for client_index, client_distribution_ in enumerate(p_client_distribution):
                    target_index_distribution[client_index] = []
                    for class_index, class_percentage in enumerate(client_distribution_):
                        num_given_samples = int(class_percentage*class_avaliable_samples[class_index])
                        target_index_distribution[client_index] += list(class_indices[class_index][:num_given_samples])
                        class_indices[class_index] = class_indices[class_index][num_given_samples:]    
                for client_index in range(num_clients):
                    target_index_distribution[client_index] = np.array(target_index_distribution[client_index])
                return target_index_distribution
            distribution = create_index_distribution(self.targets, preset_global_distribution)
        else:
            #batch_idx defines which client gets which samples by idx
            if self.distribution_config.partition_mode in ['homo', 'homogenous', 'iid', 'IID']:
                idx = np.random.permutation(N)
                batch_idx = np.array_split(idx, num_clients)
            elif self.distribution_config.partition_mode in ['hetereo_dirichlet']:
                partitioner = CIFAR10Partitioner(self.dataset.targets, num_clients, balance=None, partition='dirichlet', dir_alpha=self.distribution_config.dirichlet_alpha, seed=self.seed)
                partitioner.num_classes = num_classes
                batch_idx = partitioner._perform_partition()
            elif self.distribution_config.partition_mode in ['disjoint_shards']: # FLT paper scenario 1
                batch_idx = disjoint_shards_split(self.dataset.targets, num_clients, self.distribution_config.num_varying_clusters, self.distribution_config.num_classes_per_client, self.seed)
            elif self.distribution_config.partition_mode in ['overlapping_shards']: # FLT paper scenario 2
                batch_idx = overlapping_shards_split(self.dataset.targets, num_clients, self.distribution_config.num_varying_clusters, self.distribution_config.num_classes_per_client, self.seed) 
                # NOTE the original implementation does create overlapping client datasets, which should not be tolerated! We make sure that this does not happen..
            else:
                raise NotImplementedError(f'Partition mode {self.distribution_config.partition_mode} is not implemented yet!')
            assert len(batch_idx) == num_clients, f'Number of clients ({len(batch_idx)}) does not match the number of clients in the client_list ({num_clients})!'
            # concat all the indices of the clients into one list
            check_indices = np.concatenate([batch_idx[i] for i in range(num_clients)])
            assert len(np.unique(check_indices)) == len(check_indices), 'There are overlapping indices between clients!'
            distribution = {client: batch_idx[i] for i, client in enumerate(self.client_list)}
        torch.save(distribution, self.distribution_dump_path)
        print(f'Saved distribution to {self.distribution_dump_path}')
        return distribution
