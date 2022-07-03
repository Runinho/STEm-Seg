import os
from pathlib import Path
from typing import Dict, Union, List

import yaml


class YamlConfig(dict):
    def __init__(self, d: Dict, scope):
        self.__immutable = False
        self.__scope = scope
        super(self.__class__, self).__init__()

        for k, v in d.items():
            if isinstance(v, dict):
                self.__setattr__(k, self.__class__(v, self.__scope + k + '.'))
            else:
                self.__setattr__(k, v)

        self.__immutable = True  # prevent changes at runtime

    @property
    def scope(self):
        return self.__scope

    def __getattr__(self, item):
        attr = self.get(item, None)
        if attr is None and not item.startswith('_' + self.__class__.__name__ + '__'):
            raise ValueError("No attribute named '%s' found in config scope '%s'" % (item, self.__scope))
        return attr

    def enforce_immutable(self, key):
        print(key)
        # whitelist __immutable
        if key == '_' + self.__class__.__name__ + '__immutable':
            return

        if self.__immutable:
            raise ValueError("The config is immutable and cannot be modified")

    def allow_immutable(func):
        """decorator to override immutability"""
        def call(self, *args, **kwargs):
            self.__immutable = False
            ret = func(self, *args, **kwargs)
            self.__immutable = True
        return call

    def __setattr__(self, key, value):
        self.enforce_immutable(key)
        return self.__setitem__(key, value)

    def __setitem__(self, key, value):
        self.enforce_immutable(key)
        return super(self.__class__, self).__setitem__(key, value)

    def __str__(self):
        return self.pretty()

    def __repr__(self):
        return self.pretty()

    def pretty(self, left_margin=0):
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
    def merge_with(self, opts:Union[Dict, "YamlConfig"], strict=True, verbose=False) -> List:
        unexpected_keys = []

        for key, val in opts.items():
            if key.startswith("_YamlConfig__"):
                continue

            if key not in self:
                if strict:
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

    def merge_from_file(self, path: Path, strict=True, verbose=False):
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
        """
        Update the values based on user input given via 'argparse.ArgumentParser'.
        :param args:
        :param verbose:
        :param prefix: If the arg names have some prefix attached to them, provide it here so it is parsed correctly.
        :return:
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
        """
        Populates an ArgumentParser instance with argument names from the config instance.
        :param parser: Instance of argparse.ArgumentParser
        :param recursive: If True, config values in nested scoped will also be added
        :param prefix: A string prefix that will be prepended to the arg names
        :return:
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
                parser.add_argument('--{}{}'.format(prefix_, key.lower()), nargs='*', type=type(val[0]), required=False)
            elif isinstance(val, bool):
                parser.add_argument('--{}{}'.format(prefix_, key.lower()), type='bool', required=False)
            else:
                parser.add_argument('--{}{}'.format(prefix_, key.lower()), type=type(val), required=False)

        return parser

    def d(self) -> Dict:
        """
        Converts the object instance to a standard Python dict
        :return: object instance parsed as dict
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
    def load_from_file(cls, config_file_path: Path) -> YamlConfig:
        assert config_file_path.exists(), "config file not found at given path: %s" % config_file_path

        # check if we need to use yaml.FullLoader
        pyyaml_major_version = int(yaml.__version__.split('.')[0])
        pyyaml_minor_version = int(yaml.__version__.split('.')[1])
        required_loader_arg = pyyaml_major_version >= 5 and pyyaml_minor_version >= 1

        with open(config_file_path, 'r') as readfile:
            if required_loader_arg:
                d = yaml.load(readfile, Loader=yaml.FullLoader)
            else:
                d = yaml.load(readfile)

        yaml_config = cls(d, '')
        return yaml_config


# Init global cfg object with the values in defaults.yaml
cfg = YamlConfig.load_from_file(Path(__file__).parent / 'defaults.yaml')