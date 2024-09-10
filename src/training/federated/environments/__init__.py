from .fedavg import SiloFederatedAveragingEnvironment
from .hierarchical.fedavg import FedAvgClustersEnvironment
from .hierarchical.isolated_clusters import IsolatedClustersEnvironment

def get_federated_environment(environment_name, fed_dataset, config, *args, **kwargs):
    # SIMPLE CENTRAL, NON HIERARCHICAL ENVIRONMENTS
    if environment_name == 'SiloFederatedAveragingEnvironment':
        env = SiloFederatedAveragingEnvironment(fed_dataset, config)

    # HIERARCHICAL ENVIRONMENTS             
    elif environment_name == 'FedAvgClustersEnvironment':
        env = FedAvgClustersEnvironment(fed_dataset, config)
    elif environment_name == 'IsolatedClustersEnvironment':
        env = IsolatedClustersEnvironment(fed_dataset, config)
    else:
        raise NotImplementedError(f'{environment_name} is not implemented yet..')
    return env