import os
from collections import defaultdict
import warnings

import torch
import numpy as np
import torchmetrics
from torchmetrics import Metric
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.toolkit.config import yaml_to_munch


IMPLEMENTED_METRICS = ['Accuracy', 'SKAccuracy', 'BalancedAccuracy',]




def get_metric_function(metric, *args, **kwargs):
    """
    Get the metric function based on the provided metric configuration.

    This function initializes and configures a metric function based on the provided metric
    configuration. If a string is provided, it attempts to load the metric configuration
    from the local configurations. It then checks if the metric is implemented and either
    uses a pre-defined metric from torchmetrics or evaluates the metric using a custom
    implementation.

    Args:
        metric (str or Munch): 
            The metric to be used. It can be either a string specifying the metric name
            or a Munch object containing the metric configuration.
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

    Returns:
        Callable: 
            The initialized metric function.

    Raises:
        AssertionError: 
            If the specified metric is not implemented or checked.

    Example:
        metric_function = get_metric_function(metric, *args, **kwargs)

    Note:
        - `metric` can be either a string or a Munch based metric configuration.

    """
    if isinstance(metric, str): #if only a string was given try to fetch the metric config from local configurations
        # find 'metric_config_paths.yaml in the directory of this source file
        # and load the metric config from there
        metric_paths = yaml_to_munch(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'metric_configs', '_paths.yaml'), verbose=False)
        metrics_dict = yaml_to_munch(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'metric_configs', metric_paths[metric]+'.yaml'), verbose=False)
        metric = metrics_dict.metrics[metric]
        
    assert metric.name in IMPLEMENTED_METRICS, f'The metric function {metric.name} is not yet implemented or checked.'
    metric_args = metric.metric_args if metric.metric_args else {} #because some torch metrics dont need any arguments
    if metric.own_implementation:
        metric_function = eval(metric.name)(**metric_args, **kwargs)
    else:
        metric_function = eval('torchmetrics.'+metric.tm_name)(**metric_args, **kwargs)
    
    #add functionality to the metric function
    if metric.display_name:
        metric_function.name = metric.display_name
    else:      
        metric_function.name = metric.name
    metric_function.format = metric.format
    metric_function.subject_to = metric.subject_to

    #print(f'Using {metric_function.name} as metric function.')
    if 'device' in kwargs:
        metric_function.to(kwargs['device'])
    return metric_function

#implement own Accuracy as torchmetric, which uses sklearn as base, since original torchmetric requires you to provide the number of classes in beforehand and we dont want to do that in every case
class SKAccuracy(Metric):
    """
    Accuracy metric using sklearn as base.

    This metric calculates the balanced accuracy using sklearn's `accuracy_score`.
    It extends the `Metric` class from torchmetrics.

    Args:
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

    Attributes:
        is_differentiable (bool): 
            Indicates if the metric is differentiable. (False)
        higher_is_better (bool): 
            Indicates if higher values are better. (True)
        full_state_update (bool): 
            Indicates if the full state should be updated. (True)

    Example:
        acc = SKAccuracy()
        acc.update(predictions, targets)
        acc_score = acc.compute()

    Note:
        - This metric is not differentiable.
        - It extends the `Metric` class and uses `accuracy_score` from sklearn.

    See Also:
        - `sklearn.metrics.accuracy_score` for details on accuracy calculation.

    """
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = True
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.add_state('y_true', default=[], dist_reduce_fx='cat')
        self.add_state('y_pred', default=[], dist_reduce_fx='cat')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target = target.clone().detach().cpu().numpy()
        preds = preds.clone().detach().cpu().numpy()
        #transform to categotical if one-hot encoded
        if len(target.shape) > 1:
            target = np.argmax(target, axis=1)
        if len(preds.shape) > 1:
            preds = np.argmax(preds, axis=1)
        self.y_true.extend(target)
        self.y_pred.extend(preds)

    def compute(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return torch.Tensor((accuracy_score(self.y_true, self.y_pred),)).to(self.device)  


#implement own Balanced Accuracy as torchmetric, which uses sklearn as base, since original torchmetric BACC assumes accuracy of 0 if a class is not represented in the targets/predictions
class BalancedAccuracy(Metric):
    """
    Balanced Accuracy metric using sklearn as base.

    This metric calculates the balanced accuracy using sklearn's `balanced_accuracy_score`.
    It extends the `Metric` class from torchmetrics.

    Args:
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

    Attributes:
        is_differentiable (bool): 
            Indicates if the metric is differentiable. (False)
        higher_is_better (bool): 
            Indicates if higher values are better. (True)
        full_state_update (bool): 
            Indicates if the full state should be updated. (True)

    Example:
        balanced_acc = BalancedAccuracy()
        balanced_acc.update(predictions, targets)
        acc = balanced_acc.compute()

    Note:
        - This metric is not differentiable.
        - It extends the `Metric` class and uses `balanced_accuracy_score` from sklearn.

    See Also:
        - `sklearn.metrics.balanced_accuracy_score` for details on balanced accuracy calculation.

    """
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = True
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.add_state('y_true', default=[], dist_reduce_fx='cat')
        self.add_state('y_pred', default=[], dist_reduce_fx='cat')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target = target.clone().detach().cpu().numpy()
        preds = preds.clone().detach().cpu().numpy()
        #transform to categotical if one-hot encoded
        if len(target.shape) > 1:
            target = np.argmax(target, axis=1)
        if len(preds.shape) > 1:
            preds = np.argmax(preds, axis=1)
        self.y_true.extend(target)
        self.y_pred.extend(preds)

    def compute(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return torch.Tensor((balanced_accuracy_score(self.y_true, self.y_pred),)).to(self.device)
