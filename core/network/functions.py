import glob
import os
import pickle

from core.utils.config import CONFIG, game_name
from core.utils.types import EnvName


def read_best_index(env_name: EnvName = game_name) -> int:
    """读取保存的最佳模型index，找不到的话返回最新版本，仍找不到返回-1"""
    path = os.path.join(CONFIG['data_dir'], env_name, CONFIG['best_index_name'])
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    # 自动降级：如果没有best_index.pkl，则寻找最大的checkpoint号
    return read_latest_index(env_name)


def read_latest_index(env_name: EnvName = game_name) -> int:
    """读取保存的最新模型index，找不到的话返回-1"""
    indices = list_all_indices(env_name)
    return max(indices) if indices else -1


def list_all_indices(env_name: EnvName = game_name) -> list[int]:
    """列出所有保存的模型index，支持 model_x.pt 和 checkpoint_x.pt"""
    pattern = os.path.join(CONFIG['data_dir'], env_name, '*.pt')
    model_files = glob.glob(pattern)
    indices = []
    for f in model_files:
        try:
            # 支持 "model_12.pt" 或 "checkpoint_12.pt"
            base = os.path.basename(f)
            index = int(base.replace('.pt', '').split("_")[-1])
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


