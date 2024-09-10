import torch
from torchvision.transforms import *


def get_transforms(transforms_dict={}, norm_parameters=None):
    """
    Generate a composition of PyTorch image transformations.

    Args:
        transforms_dict (dict): A dictionary where keys are the names of torchvision.transforms transformations
            (e.g., 'Resize', 'Normalize') and values are tuples of parameters for the transformation.
        norm_parameters (tuple or None): Parameters for normalization. Used when 'Normalize' transform is specified.

    Returns:
        torchvision.transforms.Compose or None: A composition of transformations, or None if no transforms are specified.
    """

    trans_list = []
    if 'force_none' in transforms_dict:
        trans_list = []
    elif len(transforms_dict.keys()) > 0:
        for transform, parameters in transforms_dict.items():
            #print(transform, parameters)
            if transform == 'Normalize':
                if len(parameters) == 0:
                    parameters = norm_parameters
                assert parameters is not None, f'Normalization parameters are not specified!'
            t = eval(f'{transform}(*parameters)')
            if transform == 'Grayscale': # make sure to grayscale first
                trans_list.insert(0,t)
            else:
                trans_list.append(t)
    else:
        trans_list.append(ToTensor())
    return Compose(trans_list) if len(trans_list) > 0 else None



def Grayscale_to_RGB(*args, **kwargs):
    """
    Convert grayscale images to RGB format.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torchvision.transforms.Lambda: A lambda function that repeats input channels to convert grayscale to RGB.
    """
    return Lambda(lambda x: x.repeat([3, 1, 1], 0) if x.shape[0] == 1 else x)
