from copy import deepcopy

from src.training.federated.clusters.bases import ClientCluster
from src.training.federated.environments.aggregation import fedavg_aggregation

class FedAvgCluster(ClientCluster):
    """
    Represents a cluster of clients (see ClientCluster) that uses simple FederatedAveraging for the aggregation of local client models in each cluster communication round.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate(self, local_weights, *args, **kwargs):
        return deepcopy(fedavg_aggregation(local_weights, *args, **kwargs))