import yaml
from pathlib import Path
from utils.yaml.constructors import constructors
from utils.yaml.representers import representers


def load(path, construct='tag'):
    Loader = yaml.SafeLoader
    for tag, constructor in constructors[construct].items():
        Loader.add_constructor(tag, constructor)
    try:
        if path is None:
            return None
        with open(Path(path), 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
        return data
    except (IsADirectoryError,) as err:
        return err

def save(path: Path, data, represent='tag'):
    Dumper = yaml.SafeDumper
    for datatype, representer in representers[represent].items():
        Dumper.add_representer(datatype, representer)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as stream:
        yaml.dump(data, stream, Dumper=Dumper, sort_keys=False)


def load_yaml(yaml_str, construct='tag') -> dict:
    r"""yaml string to tagged (or custom) data"""
    Loader = yaml.SafeLoader
    for tag, constructor in constructors[construct].items():
        Loader.add_constructor(tag, constructor)
    return yaml.load(yaml_str, Loader=Loader)

def dump_data(data, represent='tag') -> str:
    r"""data to tagged yaml string"""
    Dumper = yaml.SafeDumper
    for datatype, representer in representers[represent].items():
        Dumper.add_representer(datatype, representer)
    return yaml.dump(data, Dumper=Dumper, sort_keys=False)


def str2tag(yaml_str) -> dict:
    return load_yaml(yaml_str, construct="tag")

def str2obj(yaml_str) -> dict:
    return load_yaml(yaml_str, construct="obj")

def tag2str(cfg) -> str:
    return dump_data(cfg, represent="tag")

def tag2obj(cfg) -> dict:
    return str2obj(tag2str(cfg))

def obj2str(data) -> str:
    return dump_data(data, represent="obj")

