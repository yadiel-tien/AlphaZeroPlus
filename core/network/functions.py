import glob
import os
import pickle

from core.utils.config import CONFIG, game_name
from core.utils.types import EnvName


def read_best_index(env_name: EnvName = game_name) -> int:
    """读取保存的最佳模型index，找不到的话返回-1"""
    path = os.path.join(CONFIG['data_dir'], env_name, CONFIG['best_index_name'])
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return -1


def read_latest_index(env_name: EnvName = game_name) -> int:
    """读取保存的最新模型index，找不到的话返回-1"""
    indices = list_all_indices(env_name)
    return max(indices) if indices else -1


def list_all_indices(env_name: EnvName = game_name) -> list[int]:
    """列出所有保存的模型index"""
    pattern = os.path.join(CONFIG['data_dir'], env_name, '*.pt')
    model_files = glob.glob(pattern)
    indices = []
    for f in model_files:
        try:
            # f like "data/Gomoku/model_12.pt"
            index = int(os.path.basename(f).split("_")[-1].split(".")[0])
            indices.append(index)
        except (ValueError, IndexError):
            continue
    return sorted(indices)


def save_best_index(best_index: int, env_name: EnvName = game_name) -> None:
    """保存最佳model的index"""
    folder = os.path.join(CONFIG['data_dir'], env_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, CONFIG['best_index_name'])
    with open(path, "wb") as f:
        pickle.dump(best_index, f)  # type: ignore


