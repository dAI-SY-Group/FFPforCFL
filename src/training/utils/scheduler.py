import types

from torch.optim import lr_scheduler

def set_scheduler(optimizer, validation_frequency, scheduler_config=None, use_val=False):
    """
    Set the learning rate scheduler for the optimizer.

    This function initializes and configures a learning rate scheduler based on the provided
    configuration.

    Args:
        optimizer (torch.optim.Optimizer): 
            The optimizer for which the scheduler will be set.
        validation_frequency (int): 
            The frequency at which model validation is performed.
        scheduler_config (Munch): 
            The configuration object for the scheduler. Includes(depending on scheduler): step_size, gamma, mode, factor, patience, milestones.
        use_val (bool, optional): 
            Flag indicating whether validation is used. Default is False.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: 
            The initialized learning rate scheduler or None if no scheduler is specified.

    Raises:
        ValueError: If the specified scheduler is not implemented.

    Example:
        scheduler = set_scheduler(optimizer, validation_frequency, scheduler_config, use_val)
    """
    if scheduler_config is None:
        return None
    else:
        name = scheduler_config.name

    if name == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, scheduler_config.step_size, scheduler_config.gamma)
        scheduler.after_val = False
    elif name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = scheduler_config.mode, factor = scheduler_config.factor, patience=scheduler_config.patience)
        scheduler.after_val = True
    elif name == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = scheduler_config.milestones, gamma = scheduler_config.gamma)
        scheduler.after_val = False
    else:
        raise ValueError(f'The scheduler {name} is not implemented yet..')

    #print(f'Using {name} as a learning rate scheduler.')
    if scheduler.after_val and validation_frequency != 1:
        print(f'You use a learning rate scheduler that only steps after each evaluation but you only evaluate after every {validation_frequency} rounds!')
    if scheduler_config.metric:
        m = scheduler_config.metric
        if scheduler_config.use_loss: m = m + '_L'
        m = 'VAL_' + m if use_val else 'TST_' + m
        scheduler.metric = m
    scheduler.lr = types.MethodType(last_lr, scheduler)
    return scheduler

def last_lr(self):
    """
    Get the last learning rate used by the scheduler.

    This method retrieves the last learning rate used by the scheduler.

    Returns:
        float: The last learning rate.

    Example:
        lr = scheduler.last_lr()

    See Also:
        - `set_scheduler` function for setting the learning rate scheduler.

    """
    return self.state_dict()['_last_lr'][0]
