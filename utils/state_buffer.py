import os
import pickle
import threading

import numpy as np
from numpy.typing import NDArray

from .config import settings, CONFIG, game_name
from .types import EnvName


class StateBuffer:
    def __init__(self, capacity: int, name: str) -> None:
        self.buffer = np.zeros((capacity,) + settings['state_shape'], dtype=np.int32)
        self.pointer = 0
        self.size = 0
        self.capacity = capacity
        self.lock = threading.Lock()
        self.name = name

    def append(self, state: NDArray) -> None:
        with self.lock:
            self.buffer[self.pointer] = state
            self.pointer = (self.pointer + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self) -> NDArray:
        with self.lock:
            index = np.random.choice(self.size)
            return self.buffer[index]

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        with self.lock:
            self.pointer = 0
            self.size = 0

    def save(self, env_name: EnvName = game_name) -> None:
        """保存buffer"""
        path = os.path.join(CONFIG['data_dir'], env_name, f'{self.name}.pkl')
        data = {
            "states": self.buffer[:self.size],
            "size": self.size,
            "pointer": self.pointer,
        }
        with open(path, "wb") as f:
            # 指定protocol=4，否则可能出现load时卡死
            pickle.dump(data, f, protocol=4)  # type: ignore
            print(f'{self.name} saved. Size{self.size}')

    def load(self, env_name: EnvName = game_name) -> None:
        """加载buffer，支持按num增量添加，优化内存使用"""
        path = os.path.join(CONFIG['data_dir'], env_name, f'{self.name}.pkl')
        if not os.path.exists(path):
            print(f"State buffer {self.name} not found at '{path}', current length: {self.size}")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.size = data['size']
            self.pointer = data['pointer']
            self.buffer[:self.size] = data['states'][:self.size]

            print(f"State buffer {self.name}  loaded successfully. Current length: {self.size}")
