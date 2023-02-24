import inspect
import os 
import time
import yaml
import logging
import argparse
import MDAnalysis as mda

from typing import Union
from pathlib import Path
from typing import Type, TypeVar
# from pydantic import BaseSettings as _BaseSettings

PathLike = Union[str, Path]
_T = TypeVar("_T")


def build_logger(debug=0):
    logger_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logger_level, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def dict_from_yaml(yml_file): 
    return yaml.safe_load(open(yml_file, 'r'))

def dict_to_yaml(dict_t, yml_file): 
    with open(yml_file, 'w') as fp: 
        yaml.dump(dict_t, fp, default_flow_style=False)

class yml_base(object): 
    def dump_yaml(self, cfg_path: PathLike) -> None: 
        dict_to_yaml(self.get_setup(), cfg_path)

def create_path(dir_type='md', sys_label=None, create_path=True): 
    """
    create MD simulation path based on its label (int), 
    and automatically update label if path exists. 
    """
    dir_path = f'{dir_type}_run'
    if sys_label: 
        dir_path = f'{dir_path}_{sys_label}'
    if create_path:
        os.makedirs(dir_path, exist_ok=True)
    return os.path.abspath(dir_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args
