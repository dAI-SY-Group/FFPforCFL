from copy import deepcopy

from src.training.federated.environments.bases import CentralizedEnvironment
from src.training.federated.environments.aggregation import fedavg_aggregation

class SiloFederatedAveragingEnvironment(CentralizedEnvironment):
    """
    Environment for Silo FederatedAveraging.

    This class extends the CentralizedEnvironment and implements the specific FedAvg aggregation method.

    Attributes:
        Inherits attributes from CentralizedEnvironment.

    Methods:
        aggregate(local_weights, *args, **kwargs):
            Aggregate local weights using Federated Averaging with Silo.

    """
    def __init__(self, fed_dataset, config, *args, **kwargs):
        print('### Initializing SiloFederatedAveragingEnvironment (START) ###')
        super().__init__(fed_dataset, config, *args, **kwargs)
        print('### Initializing SiloFederatedAveragingEnvironment (END) ###')

    def aggregate(self, local_weights, *args, **kwargs):
        return deepcopy(fedavg_aggregation(local_weights, *args, **kwargs))
