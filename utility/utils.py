import os

from typing import Dict

def ensure_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def file_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def process_dict_key(src_dict: Dict) -> Dict:
    tgt_dict = {}
    for key, val in src_dict.items():
        tgt_dict[int(key)] = val
    return tgt_dict
