import torch

def fedavg_aggregation(local_weights, *args, **kwargs):
    """
    Perform Federated Averaging (FedAvg) aggregation.

    This function aggregates the weights from multiple local models using the Federated Averaging algorithm.

    Args:
    - local_weights (list): List of tuples containing the number of local samples used for training and local model parameters.

    Returns:
    - averaged_params (dict): Averaged model parameters (weighted by the number of local samples used for training).

    """
    training_num = 0
    for idx in range(len(local_weights)):
        (sample_num, averaged_params) = local_weights[idx]
        training_num += sample_num

    (sample_num, averaged_params) = local_weights[0]
    for k in averaged_params.keys():
        for i in range(0, len(local_weights)):
            local_sample_number, local_model_params = local_weights[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params