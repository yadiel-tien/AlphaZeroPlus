import glob
import os
import pickle

from utils.config import CONFIG, game_name
from utils.types import EnvName


def read_best_index(env_name: EnvName = game_name) -> int:
    """读取保存的最佳模型index，找不到的话返回-1"""
    path = os.path.join(CONFIG['data_dir'], env_name, CONFIG['best_index_name'])
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return -1


def read_latest_index(env_name: EnvName = game_name) -> int:
    """读取保存的最新模型index，找不到的话返回-1"""
    patten = os.path.join(CONFIG['data_dir'], env_name, '*.pt')
    model_files = glob.glob(patten)
    if model_files:
        return max(
            int(f.split("_")[-1].split(".")[0])
            for f in model_files
        )
    else:
        return -1


def write_best_index(best_index: int, env_name: EnvName = game_name) -> None:
    """保存最佳model的index"""
    folder = os.path.join(CONFIG['data_dir'], env_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, CONFIG['best_index_name'])
    with open(path, "wb") as f:
        pickle.dump(best_index, f)  # type: ignore
