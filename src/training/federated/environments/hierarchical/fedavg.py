from copy import deepcopy

from src.training.federated.environments.hierarchical.bases import FederatedHierarchicalEnvironment
from src.training.federated.environments.aggregation import fedavg_aggregation

class FedAvgClustersEnvironment(FederatedHierarchicalEnvironment):
    def __init__(self, fed_dataset, config, *args, **kwargs):
        print('### Initializing FedAvgClustersEnvironment (START) ###')
        super().__init__(fed_dataset, config, *args, **kwargs)
        print('### Initializing FedAvgClustersEnvironment (END) ###')
    
    #for global aggregation do standard FedAvg
    def aggregate(self, local_weights, *args, **kwargs):
        return deepcopy(fedavg_aggregation(local_weights, *args, **kwargs))
