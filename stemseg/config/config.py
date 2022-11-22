""" Functionality to manage parameters for training, inference, datasets

Allows combining multiple sources to one configuration.

E.g a default configuration can be partial overwriten by the current experiment configuration
with the option to overwrite some parameters from the comandline.
"""
import argparse
from pathlib import Path
from typing import Dict, Union, List

import yaml

from stemseg.utils import RepoPaths


def allow_immutable(func):
    """decorator to override immutability

    Allows the annotated function to change attributes, even if `__immutable` is set to False.

    Note:
        Calling another annotated function inside `func` is not supported.
        After execution of the other function `__immutable` would be set to False.

    Args:
        func: function to decorate

    Returns:
        annotated function
    """

    def call(obj, *args, **kwargs):
        obj.__setattr__("_" + obj.__class__.__name__ + "__immutable", False)
        ret = func(obj, *args, **kwargs)
        obj.__setattr__("_" + obj.__class__.__name__ + "__immutable", True)
        return ret

    # copy docstring over for sphinx
    call.__doc__ = func.__doc__

    return call


class YamlConfig(dict):
    """ Configuration instance

    TODO: describe usage and features
        - init
        - argparse
        - combine
    """
    def __init__(self, d, scope):
        """
        Args:
            d(Dict): dict to populate with
            scope(str) :
        """
        super().__init__()
        self.__immutable = False
        self.__scope = scope

        for k, v in d.items():
            if isinstance(v, dict):
                self.__setattr__(k, self.__class__(v, self.__scope + k + '.'))
            else:
                self.__setattr__(k, v)

        self.__immutable = True  # prevent changes at runtime

    @property
    def scope(self):
        """scope of the configuration"""
        return self.__scope

    def __getattr__(self, item):
        attr = self.get(item, None)
        if attr is None and not item.startswith('_' + self.__class__.__name__ + '__'):
            raise ValueError(
                "No attribute named '%s' found in config scope '%s'" % (item, self.__scope))
        return attr

    def _enforce_immutable(self, key):
        # whitelist __immutable
        if key in ['__immutable', '_' + self.__class__.__name__ + '__immutable']:
            return

        if self.__immutable:
            raise ValueError("The config is immutable and cannot be modified")

    def __setattr__(self, key, value):
        self._enforce_immutable(key)
        return self.__setitem__(key, value)

    def __setitem__(self, key, value):
        self._enforce_immutable(key)
        return super(self.__class__, self).__setitem__(key, value)

    def __str__(self):
        return self.pretty()

    def __repr__(self):
        return self.pretty()

    def pretty(self, left_margin=0) -> str:
        """pretty print configuration

        Args:
            left_margin(int): number of whitespace to prepend. (used in recursive calls)

        Returns:
            str: pretty printed configuration
        """
        s = ""
        for k, v in self.items():
            if k.startswith('_' + self.__class__.__name__ + '__'):
                continue

            for i in range(left_margin):
                s += " "

            if isinstance(v, self.__class__):
                s = s + k + ":\n" + str(v.pretty(left_margin + 2))
            else:
                s = s + k + ": " + str(v) + "\n"
        return s

    @allow_immutable
    def merge_with(self, other, strict=True, verbose=False) -> List:
        """
        merge_with(self, other, strict=True, verbose=False) -> List
        merge with another configuration

        copies all keys that are in `opts` to this configuration.
        If strict is True only keys that are already in this configuration are allowed.

        Args:
            other (Union[Dict, YamlConfig]): another configuration to kopy the values from
            strict (bool): if set to True does not allow to have keys in `other` that
                           are not already in this configuration

        Returns:
            unexpected keys that were in `other` but not already in this configuration
        """
        unexpected_keys = []

        for key, val in other.items():
            if key.startswith("_YamlConfig__"):
                continue

            if key not in self:
                if strict:
                    # TODO: the annotated function should handle this.
                    self.__immutable = True
                    raise ValueError("No option named '%s' exists in YamlConfig" % key)
                else:
                    unexpected_keys.append(key)
            else:
                value = self[key]
                if isinstance(value, self.__class__):
                    unexpected_keys.extend(value.merge_with(val, strict))
                else:
                    self[key] = val

        return unexpected_keys

    def merge_from_file(self, path: Path, strict=True, verbose=False) -> List:
        """merge with another configuration stored in a file

        Args:
            path(Path): path to load configuration yaml from
            strict: if True does not allow keys that are not already contained
            verbose: if

        Returns:
            List: unexpected keys that are not already in this configuration
                  but were in the loaded one
        See Also:
            :func:`merge_with`
        """
        other_cfg = self.__class__.load_from_file(path)
        return self.merge_with(other_cfg, strict=strict, verbose=verbose)

    @allow_immutable
    def update_param(self, name, new_value):
        """
        Method to update the value of a given parameter.
        :param name:
        :param new_value:
        :return:
        """
        if name not in self:
            raise ValueError("No parameter named '{}' exists".format(name))
        self.__immutable = False
        self[name] = new_value
        self.__immutable = True

    @allow_immutable
    def update_from_args(self, args, verbose=False, prefix=''):
        """Update the values based on user input given via 'argparse.ArgumentParser'.

        Args:
            args (argparse.ArgumentParser): argparse to use
            verbose (bool): if true prints some debug information
            prefix(str): prefix to remove from the arguments in argparse

        """
        for arg_name, v in vars(args).items():
            if v is None:
                continue

            arg_name = arg_name.lower().replace('-', '_')
            n_skip = len(prefix) + 1 if prefix else 0
            arg_name = arg_name[n_skip:]

            for k in self:
                if k.lower() == arg_name:
                    self[k] = v
                    if verbose:
                        print("{}{} --> {}".format(self.__scope, k, v))

    def add_args_to_parser(self, parser, recursive=False, prefix=''):
        """register arguments to an ArgumentParser instance.

        Args:
            parser(argparse.ArgumentParser) : argparser to populate
            recursive(bool): If True, config values in nested scoped will also be added
            prefix(str): A string prefix that will be prepended to the arg names

        Returns:
            modified `parser`
        """

        def str2bool(v):
            if v.lower() in ("yes", "true", "on", "t", "1"):
                return True
            elif v.lower() in ("no", "false", "off", "f", "0"):
                return False
            else:
                raise ValueError("Failed to cast '{}' to boolean type".format(v))

        parser.register('type', 'bool', str2bool)

        for key, val in self.items():
            if key.startswith('_' + self.__class__.__name__ + '__'):
                continue

            if isinstance(val, self.__class__):
                if recursive:
                    val.add_args(parser, True, prefix + self.__scope)
                else:
                    continue

            prefix_ = prefix + "_" if prefix else ""
            if isinstance(val, (list, tuple)):
                parser.add_argument('--{}{}'.format(prefix_, key.lower()), nargs='*',
                                    type=type(val[0]), required=False)
            elif isinstance(val, bool):
                parser.add_argument('--{}{}'.format(prefix_, key.lower()), type='bool',
                                    required=False)
            else:
                parser.add_argument('--{}{}'.format(prefix_, key.lower()), type=type(val),
                                    required=False)

        return parser

    def d(self) -> Dict:
        """
        Converts the object instance to a standard Python dict

        Returns:
            object instance parsed as dict
        """
        d = dict()
        for k, v in self.items():
            if k.startswith('_' + self.__class__.__name__ + '__'):  # ignore class variables
                continue
            if isinstance(v, self.__class__):
                d[k] = v.d()
            else:
                d[k] = v

        return d

    @classmethod
    def load_from_file(cls, config_file_path: Path) -> "YamlConfig":
        """ load configuration from a yaml file

        Args:
            config_file_path: path to the yaml file

        Returns:
            configuration instance.
        """
        assert config_file_path.exists(), \
            "config file not found at given path: %s" % config_file_path

        # check if we need to use yaml.FullLoader
        pyyaml_major_version = int(yaml.__version__.split('.')[0])
        pyyaml_minor_version = int(yaml.__version__.split('.')[1])
        # greater or equal than version 5.1
        required_loader_arg = (pyyaml_major_version == 5 and pyyaml_minor_version >= 1) or (pyyaml_major_version > 5)

        with open(config_file_path, 'r') as readfile:
            if required_loader_arg:
                d = yaml.load(readfile, Loader=yaml.FullLoader)
            else:
                d = yaml.load(readfile)

        yaml_config = cls(d, '')
        return yaml_config


def load_global(name):
    """overwrite the global cfg with the default config and then load the config in `name`"""
    global cfg
    # reload with defaults
    # TODO: relaoding of the default values is not realy working i don't know why though :(
    #cfg = YamlConfig.load_from_file(Path(__file__).parent / 'defaults.yaml')
    # load config
    cfg_path = RepoPaths.configs_dir() / name
    cfg.merge_from_file(cfg_path)

# Init global cfg object with the values in defaults.yaml
cfg = YamlConfig.load_from_file(Path(__file__).parent / 'defaults.yaml')
