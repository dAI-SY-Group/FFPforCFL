import os

import torch

from src.training.utils.early_stopping import EarlyStopping
from src.models.modelzoo import get_model


class ClientCluster:
    """
    Represents a cluster of clients in a hierarchical federated learning setting.

    Attributes:
        cluster_id (str or int): Identifier for the cluster.
        clients (dict): Dictionary of clients in the cluster, where keys are client IDs and values are Client objects.
        config (Config): Configuration settings for the experiment (especially config.training).
        cluster_size (int): Number of clients in the cluster.
        cluster_trn_samples (int): Total number of training samples across all clients in the cluster.
        cluster_val_samples (int): Total number of validation samples across all clients in the cluster.
        cluster_tst_samples (int): Total number of test samples across all clients in the cluster.
        cluster_model (dict): State of the 'clusterwise' global model.
        file_identifier (str): Identifier for files associated with the cluster.
        checkpoint_path (str): Path for saving checkpoints for this cluster.
        total_cluster_communication_rounds (int): Total number of communication rounds in the cluster.
        num_exchanged_models_clients (int): Total number of models exchanged by clients in the cluster.
        one_round_num_exchanged_models (int): Number of models exchanged by clients in one communication round.
        early_stopper (EarlyStopping): Early stopping mechanism for the cluster.

    Methods:
        get_sample_number(split): Get the total number of samples in a specified split (for the whole cluster, i.e. all clients in the cluster).
        train(w_global, global_communication_round): Train the cluster using global parameters.
        cluster_train_step(*args, **kwargs): Perform a training step for the cluster.
        save(path=None, best=False): Save the cluster model state.
        load(path=None, best=False): Load the cluster model state.
    """
    def __init__(self, cluster_id, clients, config, *args, **kwargs):
        print(f'### Initializing ClientCluster {cluster_id} (START) ###')
        self.cluster_id = cluster_id
        self.clients = {client.client_id: client for client in clients}
        self.config = config
        
        self.cluster_size = len(clients)
        self.cluster_trn_samples = sum([client.get_sample_number('TRN') for client in clients])
        self.cluster_val_samples = sum([client.get_sample_number('VAL') for client in clients])
        self.cluster_tst_samples = sum([client.get_sample_number('TST') for client in clients])

        self.cluster_model = get_model(self.config.model.name, self.config, return_base_model=True).state_dict()

        self.file_identifier = f'{self.config.experiment_name}_CLUSTER{self.cluster_id}'+self.config.debug_file_suffix
        self.checkpoint_path = os.path.join(self.config.checkpoint_path, self.file_identifier)
        
        self.total_cluster_communication_rounds = 0
        self.num_exchanged_models_clients = 0
        self.one_round_num_exchanged_models = 2 * self.cluster_size * self.config.training.clusters.communication_rounds if self.cluster_size > 1 else 0

        # cluster early stopping is maintained and keeps track in the hierarchical environment during evaluation!!
        if self.config.training.clusters.early_stopping:
            print('Initializing cluster EarlyStopper.')
            self.early_stopper = EarlyStopping(config.training.clusters.early_stopping.patience, config.training.clusters.early_stopping.delta, config.training.clusters.early_stopping.metric, config.training.clusters.early_stopping.use_loss, config.training.clusters.early_stopping.subject_to, self.config.data.use_val, config.training.clusters.early_stopping.verbose)
        else:
            self.early_stopper = None
        print(f'### Initializing ClientCluster {cluster_id} (END) ###')

    def get_sample_number(self, split):
        if split == 'TRN':
            return self.cluster_trn_samples
        elif split == 'VAL':
            return self.cluster_val_samples
        elif split == 'TST':
            return self.cluster_tst_samples
        else:
            raise ValueError(f'Unknown split {split}')

    def train(self, w_global, global_communication_round, *args, **kwargs):
        """
        Train the cluster using the provided global parameters.
        Do training.cluster.communication_rounds cluster_training steps.

        Args:
            w_global: Global model parameters.
            global_communication_round: The current global communication round.
            *args, **kwargs: Additional arguments.

        Returns:
            torch.nn.Module: Updated cluster model.
        """
        self.cluster_model = w_global

        for communication_round in range(self.config.training.clusters.communication_rounds):
            self.total_cluster_communication_rounds += 1
            self.cluster_train_step()
            if self.config.debug and communication_round >= 2:
                break
        
        self.num_exchanged_models_clients += self.one_round_num_exchanged_models
        return self.cluster_model


    def cluster_train_step(self, *args, **kwargs):
        """
        Perform a training step for the cluster.
        Each client does one local client training (potentially multiple epochs) and returns updated local weights. 
        These are then aggregated (self.aggregate) to update the cluster model.

        Args:
            *args, **kwargs: Additional arguments for compatibility with other training methods.
        """
        local_weights = []
        for client_id, client in self.clients.items():
            client.sent_models += 1
            client.received_models += 1
            w_local = client.train(self.cluster_model, self.total_cluster_communication_rounds) 
            if self.config.training.weigh_sample_quantity:
                local_weights.append((client.get_sample_number('TRN'), w_local))
            else:
                local_weights.append((1, w_local))

        # update cluster model weights
        self.cluster_model = self.aggregate(local_weights)

    def save(self, path=None, best=False):
        path = self.checkpoint_path if path is None else path
        if best:
            path = path + '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        torch.save(self.cluster_model, statepath)

    def load(self, path=None, best=False):
        path = self.checkpoint_path if path is None else path
        if best:
            path = path + '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        self.cluster_model = torch.load(statepath)
