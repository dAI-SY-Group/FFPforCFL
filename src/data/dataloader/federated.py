import os

from torch.utils.data import DataLoader

from src.data.dataloader.central import get_central_dataloader
from src.data.utils import split_dataset, get_fed_distribution_identifier
from src.data.transforms import get_transforms
from src.data.datasets.FederatedDataset import FederatedDataset, ClientDataset
from src.data.fingerprint import create_distribution_fingerprints

class FederatedDataloaders:
    """
    Class for managing federated dataloaders.

    Parameters:
        trn_loaders (dict): A dictionary of training dataloaders, with client IDs as keys.
        tst_loaders (dict): A dictionary of testing dataloaders, with client IDs as keys.
        val_loaders (dict): A dictionary of validation dataloaders, with client IDs as keys.
        config (FederatedConfig): A configuration object for the federated learning setup.

    Attributes:
        trn_loaders (dict): Dictionary of training dataloaders.
        tst_loaders (dict): Dictionary of testing dataloaders.
        val_loaders (dict): Dictionary of validation dataloaders.
        fed_distribution_identifier (str): Identifier for the federated distribution.
        fed_distribution_path (str): Path to the saved federated distribution file.
        fingerprint_identifier (str): Identifier for the fingerprint data, if enabled.
        fingerprint_dir (str): Directory for storing fingerprint data.
        fingerprint_path (str or None): Path to the saved fingerprint data, if enabled.
    """
    def __init__(self, trn_loaders, tst_loaders, val_loaders, config):
        self.trn_loaders = trn_loaders
        self.tst_loaders = tst_loaders
        self.val_loaders = val_loaders
        dist_config = config.data_distribution_config
        self.fed_distribution_identifier = get_fed_distribution_identifier(list(trn_loaders.keys()), dist_config, config.data_seed)
        self.fed_distribution_path = os.path.join(dist_config.path, f'{self.fed_distribution_identifier}.tdump')
        if dist_config.fingerprint:
            self.fingerprint_identifier = self.fed_distribution_identifier+f'_{dist_config.fingerprint.generator_mode}_{dist_config.fingerprint.data_mode}'
            if dist_config.fingerprint.data_mode == 'features':
                self.fingerprint_identifier += f'_{dist_config.fingerprint.feature_extractor}'
            self.fingerprint_dir = os.path.join(config.dataset_path, 'fingerprints')
            if not os.path.exists(self.fingerprint_dir):
                os.makedirs(self.fingerprint_dir)
            self.fingerprint_path = os.path.join(self.fingerprint_dir, f'{self.fingerprint_identifier}_FPs.tdump')

            assert dist_config.fingerprint.generator_mode is not None, 'No fingerprint generator mode specified!'
            assert dist_config.fingerprint.data_mode is not None, 'No fingerprint data mode specified!'
            create_distribution_fingerprints(self.trn_loaders, self.fingerprint_path, dist_config.fingerprint.generator_mode, dist_config.fingerprint.data_mode, dist_config.fingerprint.feature_extractor, dist_config.fingerprint.fe_batch_size, dist_config.reload)
        else:
            self.fingerprint_path = None

    def __str__(self):
        return f'FederatedDataloaders({self.fed_distribution_identifier})'
    
    def __repr__(self):
        return f'FederatedDataloaders({self.fed_distribution_identifier})'

def get_federated_dataloader(config, trn_transformations, val_transformations):    
    """
    Generates federated data loaders based on the provided configuration.

    Args:
        config (Munch): The configuration object containing various settings.
        trn_transformations (callable): The transformations to apply to the training data.
        val_transformations (callable): The transformations to apply to the validation data.

    Returns:
        tuple: A tuple containing three dictionaries of DataLoader objects
               (`fed_trn_loader`, `fed_tst_loader`, `fed_val_loader`).

    Raises:
        AssertionError: If `client_list` is None for the 'Fundus' dataset.

    Notes:
        - For the 'Fundus' dataset, `client_list` must be specified in the configuration.
        - For other datasets, client IDs are generated based on the number of clients participating in the training.

    Example:
        >>> config = build_config('...')
        >>> trn_transforms = ...
        >>> val_transforms = ...
        >>> FederatedDataloaders = get_federated_dataloader(config, trn_transforms, val_transforms)

    """
    distribution_config = config.data_distribution_config
    distribution_config.dataset = config.data.dataset
    distribution_config.path = os.path.join(config.dataset_path, 'federated_distributions')
    if not os.path.exists(distribution_config.path):
        os.makedirs(distribution_config.path)

    #create all the actual federated data loaders
    fed_trn_loader = {}
    fed_tst_loader = {}
    fed_val_loader = {}

    if 'FedMedMNIST' in config.data.dataset:
        client_list = config.data.client_list
        split_clients = config.data.split_clients
        num_classes_per_client = config.data.num_classes_per_client
        assert client_list is not None, 'No client list specified for MixedBenchmark dataset!'
        assert split_clients is not None, 'No split client list specified for MixedBenchmark dataset!'
        assert len(client_list) == len(split_clients), 'Client list and split client list must have the same length!'
        num_clients = sum(split_clients)
        print('Start loading multiple benchmark datasets as federated dataset with {} clients ({}) that are split into ({}) clients each respectively.'.format(num_clients, client_list, split_clients))
        for num_splits, client, num_classes in zip(split_clients, client_list, num_classes_per_client):
            print('Loading data for {} clients for {}'.format(num_splits, client))
            tmp_config = config.copy()
            tmp_config.data.dataset = client
            tmp_config.data.num_classes = num_classes
            tmp_config.training.num_clients = num_splits
            tmp_fed_loaders = get_federated_dataloader(tmp_config, trn_transformations, val_transformations)
            for i in range(num_splits):
                tmp_fed_loaders.trn_loaders[i].dataset.num_classes = num_classes
                fed_trn_loader[f'{client}_{i}'] = tmp_fed_loaders.trn_loaders[i]
                tmp_fed_loaders.tst_loaders[i].dataset.num_classes = num_classes
                fed_tst_loader[f'{client}_{i}'] = tmp_fed_loaders.tst_loaders[i]
                tmp_fed_loaders.val_loaders[i].dataset.num_classes = num_classes
                fed_val_loader[f'{client}_{i}'] = tmp_fed_loaders.val_loaders[i]
                print(f'Client {client}_{i} | Trainbatches: {len(tmp_fed_loaders.trn_loaders[i]) if tmp_fed_loaders.trn_loaders[i] is not None else 0} | Testbatches: {len(tmp_fed_loaders.tst_loaders[i]) if tmp_fed_loaders.tst_loaders[i] is not None else 0} | Validationbatches: {len(tmp_fed_loaders.val_loaders[i]) if tmp_fed_loaders.val_loaders[i] is not None else 0}')
        config.data.source_client_list = client_list
        config.data.client_list = list(fed_trn_loader.keys())
        config.data.eval_client_list = list(fed_trn_loader.keys())
        config.training.num_clients = len(config.data.client_list)
    else:
        client_list = list(range(config.training.num_clients))        
        #load the central dataset as a base
        tmp_config = config.copy()
        tmp_config.data.create_validation_split = 0
        trn_loader, tst_loader, val_loader = get_central_dataloader(tmp_config, get_transforms({'force_none':True}), get_transforms({'force_none':True}))

        print(f'Splitting central dataset into federated client datasets with a {distribution_config.partition_mode} partitioning!')
        #create federated datasets from it
        distribution_config.split = 'TRN'
        fed_trn_set = FederatedDataset(trn_loader.dataset, client_list, distribution_config, None, trn_transformations, config.data_seed) if trn_loader is not None else {client: None for client in client_list}
        distribution_config.split = 'TST'
        global_preset_distribution = {client: client_dataset.target_distribution for client, client_dataset in fed_trn_set.client_datasets.items()}
        fed_tst_set = FederatedDataset(tst_loader.dataset, client_list, distribution_config, global_preset_distribution, val_transformations, config.data_seed) if tst_loader is not None else {client: None for client in client_list}
        distribution_config.split = 'VAL'
        fed_val_set = FederatedDataset(val_loader.dataset, client_list, distribution_config, global_preset_distribution, val_transformations, config.data_seed) if val_loader is not None else {client: None for client in client_list}

        print(f'Federated DataLoaders are built according to the following federated Datasets:\n  Train {fed_trn_set}\n  Test {fed_tst_set}\n  Validation {fed_val_set}')

        for client in client_list:
            trn_set = fed_trn_set[client]
            tst_set = fed_tst_set[client]
            val_set = fed_val_set[client]

            if val_set is None and config.data.create_validation_split:
                print(f'Creating validation split of {config.data.create_validation_split} from training set for client {client}')
                trn_set, val_set = split_dataset(trn_set, (1-config.data.create_validation_split, config.data.create_validation_split), shuffle = True, seed=config.data_seed, ds1_transforms=trn_transformations, ds2_transforms=val_transformations, dataset_class=ClientDataset, id=client)

            trn_set.num_classes = config.data.num_classes
            tst_set.num_classes = config.data.num_classes
            val_set.num_classes = config.data.num_classes

            trn_loader = DataLoader(trn_set, batch_size=min(config.training.batch_size, len(trn_set)), shuffle=config.data.shuffle, drop_last=False) if trn_set is not None else None
            tst_loader = DataLoader(tst_set, batch_size=min(config.training.batch_size, len(tst_set)), shuffle=False, drop_last=False) if tst_set is not None else None
            val_loader = DataLoader(val_set, batch_size=min(config.training.batch_size, len(val_set)), shuffle=False, drop_last=False) if val_set is not None else None  

            fed_trn_loader[client] = trn_loader
            fed_tst_loader[client] = tst_loader
            fed_val_loader[client] = val_loader

            print(f'Client {client} | Trainbatches: {len(trn_loader) if trn_loader is not None else 0} | Testbatches: {len(tst_loader) if tst_loader is not None else 0} | Validationbatches: {len(val_loader) if val_loader is not None else 0}')

    fed_dataloaders = FederatedDataloaders(fed_trn_loader, fed_tst_loader, fed_val_loader, config)                    
    
    return fed_dataloaders


