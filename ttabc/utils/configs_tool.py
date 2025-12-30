# -*- coding: utf-8 -*-

# Functions in this file are used to deal with any work related with the configuration of datasets, models and algorithms.
from ttabc.configs.algorithms import algorithm_defaults
from pprint import pprint

def config_hparams(config):
    """
    Populates hyperparameters with defaults implied by choices of other hyperparameters.

    Args:
        - config: namespace
    Returns:
        - config: namespace
        - scenario: NamedTuple
    """
    # prior safety check
    assert (
        config.run_type is not None
    ), "model adaptation method must be specified"

    assert (
        config.data is not None
    ), "base_data_name must be specified, either from default scenario, or from user-provided inputs."

    # register default arguments based on model_adaptation_method
    # register default arguments based on model_adaptation_method
    if config.run_type in algorithm_defaults:
        config = defaults_registry(
            config, template=algorithm_defaults[config.run_type]
        )
    else:
        print(f"Warning: Algorithm '{config.run_type}' not found in defaults registry. Using command line arguments only.")

    bool_params = ['tpt', 'cocoop', 'two_step', 'I_augmix']
    none_params = ['ctx_init', 'load', 'log_dir']

    convert_config_params(config, bool_params, none_params)

    return config

def defaults_registry(config, template: dict):
    """
    Updates the `config` namespace with values from `template`.

    - Fields in `config` that are also in `template` will be replaced by the values in `template`.
    - Fields in `template` that are not in `config` will be added to `config`.

    Args:
        config: A namespace object.
        template (dict): A dictionary containing default values.

    Returns:
        None: The `config` object is updated in place.
    """
    for key, value in template.items():
        if hasattr(config, key):
            # Replace the value in `config` with the value from `template`
            setattr(config, key, value)
        else:
            # Add the key-value pair to `config` if it doesn't exist
            setattr(config, key, value)

    return config

def str_to_bool(value):
    """
    Converts a string to a boolean value.

    Args:
        value (str): The input value to be converted.

    Returns:
        bool: True for affirmative strings, False for negative strings.
    """
    if isinstance(value, bool):
        return value  # Already a boolean
    if isinstance(value, str):
        value = value.strip().lower()
        true_values = {'true', '1', 'y', 'yes', 't', 'on'}
        false_values = {'false', '0', 'n', 'no', 'f', 'off'}
        if value in true_values:
            return True
        elif value in false_values:
            return False
    raise ValueError(f"Cannot convert value '{value}' to bool.")


def str_to_none(value):
    """
    Converts a string to None if it represents a 'None'-like value.

    Args:
        value (str): The input value to be converted.

    Returns:
        None: If the string represents a None-like value.
        str: Otherwise, return the original string.
    """
    if value is None:  # Already None
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        none_values = {'none', 'null', 'n', '', 'nil'}
        if value in none_values:
            return None
    return value  # Return original value if not None-like


def convert_config_params(config, bool_params, none_params):
    """
    Converts specific parameters in `config` to boolean or None based on their names.

    Args:
        config (namespace): The namespace containing the parameters.
        bool_params (list): A list of parameter names that should be converted to boolean.
        none_params (list): A list of parameter names that should be converted to None.
    """
    # Convert boolean parameters
    for param in bool_params:
        if hasattr(config, param):
            value = getattr(config, param)
            try:
                setattr(config, param, str_to_bool(value))
            except ValueError as e:
                print(f"Warning: {e}")

    # Convert None-like parameters
    for param in none_params:
        if hasattr(config, param):
            value = getattr(config, param)
            setattr(config, param, str_to_none(value))