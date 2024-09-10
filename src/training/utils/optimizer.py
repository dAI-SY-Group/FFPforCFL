from torch import optim

def set_optimizer(model, optimizer_config, *args, **kwargs):
    """
    Set the optimizer for the model based on the provided configuration.

    This function selects and initializes the appropriate optimizer based on the configuration
    provided. It supports various optimizer types like SGD, Adam, LBFGS, and custom optimizers.

    Recursive call when using SplitOptimizer (i.e. different parts of the model are optimized with different optimizers).

    Args:
        model (nn.Module or list): 
            The neural network model or a list of parameters if using custom optimizers.
        optimizer_config (Munch): 
            Configuration object specifying the optimizer type and parameters. Includes: name (str), lr (float), momentum (float), weight_decay (float), beta1 (float), beta2 (float), and differential_privacy (Munch).
        *args: 
            Additional positional arguments.
        **kwargs: 
            Additional keyword arguments.

    Returns:
        torch.optim.Optimizer: 
            The initialized optimizer for the model.

    Raises:
        ValueError: If the specified optimizer in the configuration is not implemented.

    Example:
        optimizer = set_optimizer(model, optimizer_config)

    Note:
        - The `optimizer_config` parameter should be an instance of `OptimizerConfig` containing
          the optimizer name and associated parameters.

    """        
    parameters = model if type(model) == list else model.parameters()
    if optimizer_config.name == 'SGD':
        momentum = optimizer_config.momentum if optimizer_config is not None else 0
        weight_decay = optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0
        optimizer = optim.SGD(parameters, lr=optimizer_config.lr, momentum=momentum, weight_decay = weight_decay)
    elif optimizer_config.name == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=optimizer_config.lr, betas=(optimizer_config.beta1, optimizer_config.beta2), weight_decay = optimizer_config.weight_decay)
    elif optimizer_config.name == 'Adam':
        optimizer = optim.Adam(parameters, lr=optimizer_config.lr, betas=(optimizer_config.beta1, optimizer_config.beta2), weight_decay = optimizer_config.weight_decay)
    else:
        raise ValueError(f'The optimizer {optimizer_config.name} is not implemented yet..')

    #print(f'Using {optimizer_config.name} optimizer with learning rate {optimizer_config.lr}')
    optimizer.lr = lambda: optimizer.param_groups[0]['lr']
    optimizer.set_lr = lambda lr: set_lr(optimizer, lr)

    return optimizer

def set_lr(optim, lr):
    """
    Set the learning rate for the optimizer.

    This function allows for dynamically changing the learning rate of the optimizer during training.

    Args:
        optim (torch.optim.Optimizer): 
            The optimizer to set the learning rate for.
        lr (float): 
            The new learning rate value.

    Example:
        set_lr(optimizer, new_lr)

    Note:
        - The `optim` parameter should be an instance of a PyTorch optimizer.
        - `lr` should be a float value representing the new learning rate.

    See Also:
        - `set_optimizer` function for setting the optimizer.

    """
    for g in optim.param_groups:
        g['lr'] = lr
    