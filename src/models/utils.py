from copy import deepcopy
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torchinfo


def attach_model_functionalities(model, config, verbose=True):
    """
    Attach various functionalities to a machine learning model.

    This function attaches several methods and attributes to the given machine learning model, enhancing its capabilities
    and providing additional functionalities. These functionalities include checkpoint management, model summaries,
    testing, parameter counting, dropout mask tracking (if applicable), bottleneck layer manipulation (if applicable),
    attackable parameter counting, and more.

    Parameters:
    - model (torch.nn.Module): The machine learning model to which functionalities will be attached.
    - config (object): A configuration object containing settings for the model and training.
    - verbose (bool, optional): If True, print informative messages during the attachment of functionalities. Default is True.

    Returns:
    - model (torch.nn.Module): The input model with additional functionalities attached.

    Attached Functionalities:
    - `model.save`: Save the model to a specified checkpoint path.
    - `model.load`: Load a model from a specified checkpoint path.
    - `model.summary`: Generate a model summary.
    - `model.test`: Test the model's performance on a given data loader.
    - `model.reset_parameters`: Reset model parameters to their initial values.
    - `model.freeze`: Freeze model layers.
    - `model.unfreeze`: Unfreeze model layers for training.
    - `model.trainable_parameters`: Count trainable parameters in the model.
    - `model.untrainable_parameters`: Count untrainable parameters in the model.
    - `model.copy`: Create a copy of the model.
    - `model._config`: Store the configuration object associated with the model.

    Note:
    - The specific functionalities attached depend on the type of model and its modules. Not all functionalities
      may be relevant or available for every model.
    - The `config` object is assumed to contain necessary settings for the model and training.

    Example Usage:
    ```python
    # Attach functionalities to a machine learning model
    enhanced_model = attach_model_functionalities(model, config)
    ```

    """
    if config.checkpoint_path:
        model.savepath = os.path.join(config.checkpoint_path, config.experiment_name)
        model.save = lambda path=model.savepath: save_model(path, model)
        model.load = lambda path=model.savepath: load_model(path, model)
        if verbose:
            print(f'Setting the models checkpoint path to {model.savepath}.state')

    model.summary = lambda shape=config.data.shape: summary(model, shape)
    model.test = lambda data_loader, metric_fns, device, verbose=True: test_model(model, data_loader, metric_fns, device, verbose)
    model.reset_parameters = lambda verbose=False: reset_parameters(model, verbose)
    model.freeze = lambda: freeze_model(model)
    model.unfreeze = lambda: unfreeze_model(model)
    model.trainable_parameters = lambda trainable=True: count_parameters(model, trainable)
    model.untrainable_parameters = lambda trainable=False: count_parameters(model, trainable)
    model.copy = lambda: copy_model(model)
    model._config = config

    return model


def get_layer(layer_str, parameters, force_parameters=False, **kwargs):
    """
    Get a neural network layer.

    This function takes a layer string (e.g., nn.ReLU, nn.Linear) and the corresponding parameters 
    required for that module.

    Args:
        layer_str (str): The string representation of the layer (e.g., 'nn.ReLU').
        parameters (list): List of parameters needed for the specified layer.
        force_parameters (bool, optional): If True, force the use of provided parameters, 
            otherwise use default parameters. Defaults to False.
        **kwargs: Additional keyword arguments that can be used to pass specific parameters 
            required by certain layers (e.g., 'channels' for BatchNorm and InstanceNorm layers).

    Returns:
        torch.nn.Module: The specified layer module.

    Raises:
        NameError: If the provided layer string is not a valid module.

    Example:
        >>> get_layer('nn.Linear', [1024, 1024])
        Linear(in_features=1024, out_features=1024, bias=True)

    Note:
        - For BatchNorm and InstanceNorm layers, if `force_parameters` is not set to True,
          it will use 'channels' from keyword arguments as parameters.

    """
    if 'BatchNorm' in layer_str or 'InstanceNorm' in layer_str and not force_parameters:
        parameters = [kwargs['channels']]
    return eval(f'{layer_str}(*parameters)')


def summary(model, input_shape=(3, 32, 32)):
    """
    Print a summary of the model architecture.

    Args:
        model (torch.nn.Module): The neural network model to summarize.
        input_shape (tuple, optional): The shape of the input data. Defaults to (3, 32, 32).

    Returns:
        torchinfo.ModelStatistics: The torchinfo model statistics object.

    Example:
        >>> model_stats = summary(MyModel(), input_shape=(3, 64, 64))

    """
    model_stats = torchinfo.summary(model, input_shape, batch_dim=0)
    print(model_stats)
    print((model_stats.total_input + model_stats.total_output_bytes + model_stats.total_param_bytes), 'Bytes')
    print((model_stats.total_input + model_stats.total_output_bytes + model_stats.total_param_bytes)/1e3, 'KB')
    return model_stats



def save_model(path, model):
    """
    Save the model to a file.

    Args:
        path (str): The file path to save the model.
        model (torch.nn.Module): The neural network model to be saved.

    Returns:
        None

    Example:
        >>> save_model('my_model.state', MyModel())

    """
    path = path if path.endswith('.state') else path + '.state'
    torch.save(model.state_dict(), path)

def load_model(path, model):
    """
    Load a model from a file.

    Args:
        path (str): The file path to load the model from.
        model (torch.nn.Module): The neural network model.

    Returns:
        torch.nn.Module: The loaded neural network model.

    Example:
        >>> loaded_model = load_model('my_model.state', MyModel())

    Raises:
        FileNotFoundError: If the specified checkpoint file is not found.

    """
    path = path if path.endswith('.state') else path + '.state'
    if os.path.isfile(path):
        load_dict = torch.load(path)
        model.load_state_dict(load_dict)
        print(f'Loaded model from {path}...')
        return model
    else:
        print(
            f'Checkpoint file {path} was not found. No other weights were loaded into the model...')
        return model

def reset_parameters(model, verbose=False):
    """
    Recursively resets the parameters of a PyTorch model and its sub-modules.

    Args:
        model (nn.Module): The PyTorch model to reset parameters for.
        verbose (bool, optional): Whether to print verbose messages. Defaults to False.
    """
    children = list(model.children())
    for child in children:
        if len(list(child.children())) > 0:
            reset_parameters(child, verbose)
        else:
            try:
                child.reset_parameters()
                if verbose:
                    print(f'Resetting parameters of {child}!')
            except:
                if verbose:
                    print(f'{child} has no parameters to be reset!')
                continue

def freeze_model(model):
    """
    Freezes all the parameters of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    """
    Unfreezes all the parameters of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True

def copy_model(model):
    """
    Create a deep copy of a machine learning model with attached functionalities.

    This function creates a deep copy of the provided machine learning model. The copy will have the same architecture
    and weights, and will also have the same set of functionalities attached.

    Parameters:
    - model (torch.nn.Module): The machine learning model to be copied.

    Returns:
    - torch.nn.Module: A deep copy of the input model with attached functionalities.

    Example Usage:
    ```python
    # Create a deep copy of a model
    copied_model = copy_model(original_model)
    ```

    """
    config = model._config
    model_copy = deepcopy(model)
    model_copy = attach_model_functionalities(model_copy, config, verbose=False)
    return model_copy
        
def test_model(model, data_loader, metric_fns, device, verbose=True):
    """
    Test the model on the given data_loader.

    Args:
        model (torch.nn.Module): The model to test.
        data_loader (torch.utils.data.DataLoader): The data loader to test the model on.
        metric_fns (list): A list of metric functions to evaluate the model on.
        device (torch.device): The device to use for the model.

    Returns:
        dict: A dictionary containing the aggregated metric values.
    """
    model.to(device)
    model.eval()
    results = {}
    if not isinstance(data_loader, list):
        data_loader = [data_loader]
    with torch.no_grad():
        for data_loader_ in data_loader:
            #dynamically use tqdm if verbose
            if verbose:
                data_loader_ = tqdm(data_loader_)
            for data, target in data_loader_:
                data, target = data.to(device), target.to(device)
                output = model(data)
                for metric_fn in metric_fns:
                    metric_fn(output, target)
    for metric_fn in metric_fns:
        results[metric_fn.name] = metric_fn.compute().item()
        metric_fn.reset()
    return results

def count_parameters(model, trainable=True):
    """
    Count the number of parameters in a machine learning model.

    This function counts the number of trainable/untrainable parameters in the provided machine learning model.

    Parameters:
    - model (torch.nn.Module): The machine learning model to count parameters in.
    - trainable (bool, optional): If True, count only trainable parameters. If False, count only untrainable parameters.
                                  Default is True.

    Returns:
    - int: The number of parameters.

    Example Usage:
    ```python
    # Count trainable parameters in a model
    num_trainable_params = count_parameters(model, trainable=True)

    # Count untrainable parameters in a model
    total_params = count_parameters(model, trainable=False)
    ```

    """
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)
