import os
import yaml
from munch import Munch, DefaultMunch

def build_config(source, debug=False, verbose=True):
    """
    Build a configuration object based on a provided source.

    Args:
        source (dict or str): Either a dictionary containing configuration information
            or the path to a YAML file containing configuration information.
        debug (bool, optional): Whether to set debug mode. Defaults to False.
        verbose (bool, optional): Whether to display verbose information. Defaults to True.

    Returns:
        Munch: A configuration object.
    """
    config = get_partial_config(source, verbose, main=True)
    if debug:
        config.debug = True
    config.debug_file_suffix = '_debug' if debug else ''
    return config

def get_partial_config(source, verbose=True, main=False):
    """
    Get a partial configuration object based on a provided source.

    Args:
        source (dict or str): Either a dictionary containing configuration information
            or the path to a YAML file containing configuration information.
        verbose (bool, optional): Whether to display verbose information. Defaults to True.
        main (bool, optional): Whether this is the main configuration. Defaults to False. Main configs require to have a project specified (e.g. main experiments or evaluation configs)

    Returns:
        Munch: A partial configuration object.
    """
    if type(source) == dict:
        main_config = DefaultMunch.fromDict(source, None)
    else:
        assert type(source) == str
        source = source if source.endswith('.yaml') else source+'.yaml'
        if verbose:
            print(f'Loading config from file: {source}')
        with open(source) as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.Loader)
        main_config = DefaultMunch.fromDict(data, None)
    config = DefaultMunch.fromDict({}, None)
    
    if main_config.base_configs is not None:
        for c_file in main_config.base_configs:
            check_file = os.path.join('configs/bases/', c_file+'.yaml')
            if os.path.exists(check_file):
                c_file = check_file
            else:
                raise ValueError(f'Base config file {c_file} does not exist!')
            partial_config = get_partial_config(c_file, verbose)
            update_config(config, partial_config)
    update_config(config, main_config)
    return config

def update_config(d, u):
    """
    Update a configuration object with values from another.

    Args:
        d (Munch): The target configuration object to be updated.
        u (Munch): The source configuration object.

    Returns:
        Munch: The updated configuration object.
    """
    try:
        for k, v in u.items():
            if isinstance(v, Munch):
                d[k] = update_config(d.get(k, DefaultMunch()), v)
            else:
                d[k] = v
    except:
        print(f'WARNING: Something went wrong when trying to update {k} with {v} of config {d}')
        assert False
    return d

def modify_config(from_file_name, to_file_name, param_change_dict, base_path, force = False):
    """
    Modify a configuration file by changing specific parameters.

    Args:
        from_file_name (str): The name of the source configuration file.
        to_file_name (str): The name of the target configuration file.
        param_change_dict (dict): A dictionary containing parameter changes.
        base_path (str): The base path for the configuration files.
        force (bool, optional): Whether to force overwrite if the target file already exists.
            Defaults to False.

    Returns:
        None
    """
    from_file_path = base_path + from_file_name + '.yaml'
    to_file_path = base_path + to_file_name + '.yaml'
    if not force:
        assert not os.path.isfile(to_file_path), f'{to_file_path} already exists!'
    with open(from_file_path) as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.Loader)
    
    data.update(param_change_dict)

    with open(to_file_path, 'w') as f:
        data_n = yaml.dump(data, f, sort_keys=False, default_flow_style = False)

def modify_config_id(config, id, value):
    """
    Modify a configuration by changing a specific parameter.

    Args:
        config (Munch): The target configuration object.
        id (str): The identifier of the parameter to be changed.
        value: The new value for the parameter.

    Returns:
        None
    """
    exec(f'config.{id} = value')

def yaml_to_munch(yaml_file, verbose=False):
    """
    Convert YAML configuration file to Munch object.

    This function reads a YAML configuration file and converts it to a Munch object,
    which is essentially a dictionary with attribute-style access.

    Args:
        yaml_file (str): 
            The path to the YAML configuration file.
        verbose (bool, optional): 
            Whether to print loading message. Default is False.

    Returns:
        DefaultMunch: 
            A Munch object containing the configuration.

    Example:
        config = yaml_to_munch('config.yaml')
        print(config.model.architecture)

    Note:
        - This function requires the `DefaultMunch` class.
        - The `yaml` package is used for parsing the YAML file.
    """
    yaml_file = yaml_file if yaml_file.endswith('.yaml') else yaml_file+'.yaml'
    if verbose:
        print(f'Loading config from file: {yaml_file}')
    with open(yaml_file) as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.Loader)
    return DefaultMunch.fromDict(data, None)
