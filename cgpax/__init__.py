import copy
from functools import reduce
from pathlib import Path
from typing import Dict, List

import yaml as yaml


def get_config(filename: str) -> Dict:
    config = yaml.safe_load(Path(filename).read_text())
    flat_config = {}
    for key in config:
        if isinstance(config[key], dict) and "value" in config[key]:
            flat_config[key] = config[key]["value"]
        else:
            flat_config[key] = config[key]
    return flat_config


# methods for configs management

def _unnest_dictionary(config: Dict, nesting_keyword: str = "nested") -> List[Dict]:
    nested_values = config[nesting_keyword]
    del config[nesting_keyword]
    return [config | x for x in nested_values]


def _unnest_dictionary_recursive(config: Dict, nesting_keyword: str = "nested") -> List[Dict]:
    nesting_keywords = [x for x in config.keys() if nesting_keyword in x]
    configs = [config]
    partially_unpacked_configs = []
    for keyword in nesting_keywords:
        for conf in configs:
            partially_unpacked_configs += _unnest_dictionary(conf, keyword)
        configs = partially_unpacked_configs
        partially_unpacked_configs = []
    return configs


def _unpack_dictionary(config: Dict) -> List[Dict]:
    config_list = []
    for key, value in config.items():
        if isinstance(value, dict):
            unpacked_value = _unpack_dictionary(value)
            if len(unpacked_value) > 1:
                config[key] = unpacked_value
    for key, value in config.items():
        if isinstance(value, list) and len(value) > 1 and not key.endswith("_list"):
            for v in value:
                temp_config = copy.deepcopy(config)
                temp_config[key] = v
                config_list += _unpack_dictionary(temp_config)
            break
    if len(config_list) == 0:
        config_list.append(config)
    return [{key.replace("_list", ""): value for key, value in cfg.items()} for cfg in config_list]


def process_dictionary(config: Dict, nesting_keyword: str = "nested") -> List[Dict]:
    config_list = _unnest_dictionary_recursive(config, nesting_keyword)
    return list(reduce(lambda x, y: x + y, [_unpack_dictionary(x) for x in config_list], []))
