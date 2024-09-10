from src.models.utils import attach_model_functionalities
from src.models.CNN import build_CNN

MODEL_ARCHITECTURES = [
                       'CNN'
                    ]


def get_model(architecture, config, verbose=True, return_base_model=False, *args, **kwargs):
    """
    Constructs a deep learning model based on the specified architecture.

    Args:
        architecture (str): Name of the desired model architecture.
        config (Config): Configuration object containing model settings.
        verbose (bool, optional): Whether to print verbose messages. Defaults to True.
        return_base_model (bool, optional): Whether to return the base model (if applicable). Defaults to False.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module or None: The constructed model or the base model (if return_base_model=True).
    """
    assert type(architecture) == str, f'config.model.name ({architecture}) must be a string!'
    if 'CNN' in architecture:
        model = build_CNN(architecture, config)
    else:
        raise NotImplementedError(f'The model architecture {architecture} is not implemented yet..')
    if verbose:
        print(f'Loaded model with {architecture} architecture, input shape {config.data.shape}, {config.num_classes} classes.')
    
    if not return_base_model:
        model = attach_model_functionalities(model, config, verbose)
    return model