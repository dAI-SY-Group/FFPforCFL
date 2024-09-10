import torch

IMPLEMENTED_LOSSES = ['CrossEntropy',]

def get_loss_function(loss, *args, **kwargs):
    """
    Returns the specified loss function.

    This function returns an instance of a loss function based on the input `loss` and model type.

    Args:
        loss (str):
            The name of the loss function to be used.
        model (nn.Module, optional):
            The neural network model for which the loss function will be applied. Default is None.
        weighted (str, optional):
            The type of weighting to be applied when considering gradient loss functions to the loss function. Default is None.
        top_k (int, optional):
            The number of top-k largest gradients to consider in the gradient loss function. Default is None.
        ignore_layers (list, optional):
            List of layers to ignore when computing the loss. Default is an empty list.

    Returns:
        loss_function (nn.Module):
            The specified loss function.

    Raises:
        ValueError: If the specified loss function is not implemented.

    Note:

    Example:
        loss_function = get_loss_function('CrossEntropy')
    """
    if loss == 'CrossEntropy':
        loss_function = CrossEntropy()
    else:
        raise ValueError(f'The loss function {loss} is not implemented yet.')
    return loss_function

class Loss:
    def __init__(self):
        self.loss_fn = None
    def __call__(self, prediction, targets):
        if self.loss_fn == None:
            raise  NotImplementedError()
        else:
            return self.loss_fn(prediction, targets)

class CrossEntropy(Loss):
    def __init__(self):
        self.loss_fn = self.ce
        self.nn_ce = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        self.format = '.5f'
        self.name = 'CrossEntropy'
        self.subject_to = 'min'
    def ce(self, prediction, target):
        if prediction.shape == target.shape:
            ce = - torch.mean(torch.sum(torch.softmax(target, -1) * torch.log(torch.softmax(prediction, -1)), dim=-1))
        else:
            ce = self.nn_ce(prediction, target)
        return ce