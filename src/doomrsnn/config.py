from pathlib import Path
import copy
import yaml


def load_config(path):
    with Path(path).open(encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError('Configuration must be a YAML mapping')
    return data


def override_config(config, **overrides):
    cfg = copy.deepcopy(config)
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val
    return cfg
