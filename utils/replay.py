import os.path
import pickle

import numpy as np
from numpy.typing import NDArray

from env.functions import get_class
from .config import game_name, CONFIG, settings
from .types import EnvName


class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, game: EnvName = game_name) -> None:
        self.state_buffer = np.zeros((capacity,) + settings['tensor_shape'], dtype=np.float32)
        self.policy_buffer = np.zeros((capacity, settings['default_net']['n_actions']), dtype=np.float32)
        self.value_buffer = np.zeros((capacity,), dtype=np.float32)
        self.batch_size = batch_size
        self.pointer = 0
        self.size = 0
        self.game = game
        self.env = get_class(game)
        self.capacity = capacity

    def add(self, data: tuple[NDArray, NDArray, float]) -> None:
        """先数据增强再添加
        :param data: (state,pi,q)
        """
        augmented_data = self.env.augment_data(data)
        for state, pi, q in augmented_data:
            self.append(state, pi, q)

    def append(self, state: NDArray, pi: NDArray, q: float) -> None:
        self.state_buffer[self.pointer] = state
        self.policy_buffer[self.pointer] = pi
        self.value_buffer[self.pointer] = q
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_batch(self) -> tuple[NDArray, NDArray, NDArray]:
        """随机采样一个 batch 进行训练
            :return (state,pi,q)
                - state:[B,C,H,W]
                - pi:[B,H]
                - q:[B]"""
        if self.size < self.batch_size:
            raise ValueError(f'No enough data! Current size is {self.size},required size is {self.batch_size}.')
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        states = self.state_buffer[indices].copy()
        probs = self.policy_buffer[indices].copy()
        values = self.value_buffer[indices].copy()
        # 复制避免引用过多，导致无法及时清理资源
        return states, probs, values

    def save(self, env_name: EnvName = game_name) -> None:
        """保存buffer"""
        path = os.path.join(CONFIG['data_dir'], env_name, CONFIG['buffer_name'])
        data = {
            "states": self.state_buffer[:self.size],
            "pis": self.policy_buffer[:self.size],
            "values": self.value_buffer[:self.size],
            "size": self.size,
            "pointer": self.pointer,
            "game": self.game
        }
        with open(path, "wb") as f:
            # 指定protocol=4，否则可能出现load时卡死
            pickle.dump(data, f, protocol=4)  # type: ignore

    def load(self, env_name: EnvName = game_name) -> None:
        """加载buffer，支持修改容量"""
        path = os.path.join(CONFIG['data_dir'], env_name, CONFIG['buffer_name'])
        if not os.path.exists(path):
            print(f"Buffer not found at '{path}', current length: {self.size}")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
            if data['game'] != self.game:
                raise ValueError('Game mismatch! Replay buffer loading Failed!')

            count = min(data['size'], self.capacity)
            src_capacity = data['states'].shape[0]
            pointer = data['pointer']
            if data['size'] < src_capacity:  # 尚未存满
                indices = np.arange(data['size'] - count, data['size'])
            else:
                # 提取逻辑上的最后 count 条数据。
                # 示例：容量10，当前指针2 (最新数据在1,0,9,8...)，需要取5条。
                # 1. 逻辑起点：pointer(2) - count(5) = -3
                # 2. 生成序列：[-3, -2, -1, 0, 1]
                # 3. 对10取模：[7, 8, 9, 0, 1] -> 正确对应了环形存储的物理索引
                indices = (np.arange(count) + (pointer - count)) % src_capacity
            self.size = count
            self.pointer = count % self.capacity

            self.state_buffer[:count] = data['states'][indices]
            self.policy_buffer[:count] = data['pis'][indices]
            self.value_buffer[:count] = data['values'][indices]
            print(f"Replay buffer loaded successfully. Current length: {self.size},Capacity: {self.capacity}")

    def clear(self) -> None:
        self.size = 0
        self.pointer = 0

    def __len__(self):
        return self.size
