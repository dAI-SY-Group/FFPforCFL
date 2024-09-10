class EarlyStopping:
    """
    Early stopping criteria for model training.
    Early stops the training if validation loss doesn't improve after a given patience.
    
    This class implements early stopping based on a specified metric.
    It monitors the metric and stops training if the metric does not improve
    for a certain number of epochs.

    Code inspired by https://github.com/Bjarten/early-stopping-pytorch 26.03.2021

    Args:
        patience (int, optional):
            Number of epochs with no improvement after which training will be stopped. Default is 7.
        delta (int, optional):
            Minimum change in the monitored metric to be considered as an improvement. Default is 0.
        metric (str, optional):
            The metric to be monitored for early stopping. Default is 'CrossEntropy'.
        use_loss (bool, optional):
            Whether to use loss as the monitored metric. Default is True.
        subject_to (str, optional):
            Whether to minimize ('min') or maximize ('max') the monitored metric. Default is 'min'.
        use_val (bool, optional):
            Whether to use validation set for monitoring the metric. Default is True.
        verbose (bool, optional):
            Whether to print early stopping information. Default is False.

    Methods:
        get_state():
            Returns the state of the EarlyStopping object.
        set_state(state_dict):
            Sets the state of the EarlyStopping object.
        __call__(metric):
            Called at each epoch to update the early stopping criteria.
        reset():
            Resets the early stopping criteria.
        soft_reset():
            Resets the early stopping criteria without removing the best score.
    """
    def __init__(self, patience=7, delta=0, metric='CrossEntropy', use_loss=True, subject_to='min', use_val=True, verbose=False):
        self.patience = patience
        self.use_loss = use_loss
        self.use_val = use_val        
        if use_loss: metric = metric + '_L'
        if use_val is not None:
            metric = 'VAL_' + metric if use_val else 'TST_' + metric
        self.metric = metric
        self.subject_to = subject_to
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.improved = False
        self.delta = delta

    def get_state(self):
        return self.__dict__

    def set_state(self, state_dict):
        self.__dict__ = state_dict

    def __call__(self, metric):
        if self.subject_to == 'max':
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.improved = True
            self.best_score = score
        elif score >= self.best_score + self.delta:
            self.improved = False
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.improved = True
            self.best_score = score
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.improved = False

    def soft_reset(self):
        self.counter = 0
        self.stop = False
        self.improved = False
                
