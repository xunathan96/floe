from pathlib import Path
from datetime import datetime
import utils.yaml.parser as parser


class Config(dict):

    def __init__(self, *args, **kwds):
        return super().__init__(*args, **kwds)
    
    def parse(self):
        """load any yaml files specified in Config"""
        self.__init__(Config._parse(self))

    @staticmethod
    def _parse(tree: dict) -> dict:
        branch = dict()
        for k,v in tree.items():
            if isinstance(v, str) and Path(v).suffix == '.yml':
                branch[k] = parser.load(v, construct='tag')
            elif isinstance(v, dict):
                branch[k] = Config._parse(v)
            else:
                branch[k] = v
        return branch

    def save(self, filename: str):
        parser.save(Path(filename).with_suffix('.yml'), dict(self), represent='tag')

    def load(self, filename: str):
        config = parser.load(Path(filename).with_suffix('.yml'), construct='tag')
        return self.__init__(config)

    def __or__(self, value: dict):
        return Config(**super().__or__(value))
    
    def __ror__(self, value):
        return Config(**super().__ror__(value))


class Experiment:

    def __init__(self, dir: str = None, **hyperparams):
        r"""Define an experiment class at the given directory."""
        self.config = Config(**hyperparams)
        self.dir = Path(dir or datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.modules = None

    def __repr__(self):
        return f"{self.config}"

    def __getitem__(self, key):
        return self.modules[key]

    def __setitem__(self, key, value):
        self.modules[key] = value

    def build(self, cfg: Config):
        r"""build custom modules based on the given config."""
        self.config = self.config | cfg
        self.modules = parser.tag2obj(dict(self.config))
        return self.modules

    def save(self, filename: str, rel_path: bool = True):
        r"""save the experiment configuration files."""
        path = self.dir / Path(filename) if rel_path else Path(filename)
        self.config.save(path)

    def load(self, filename: str, rel_path: bool = True):
        r"""load the experiment configuration files."""
        path = self.dir / Path(filename) if rel_path else Path(filename)
        return self.config.load(path)

