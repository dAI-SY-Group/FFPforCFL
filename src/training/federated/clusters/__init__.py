from .bases import ClientCluster
from .fedavg import FedAvgCluster

def get_cluster_class(cluster_type):
    return eval(cluster_type)