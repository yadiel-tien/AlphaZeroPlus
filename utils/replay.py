import os.path
import pickle

import numpy as np
from numpy.typing import NDArray

from .config import game_name, CONFIG
from .mirror import apply_symmetry, switch_side_ip, mirror_board_policy, mirror_action_policy
from .types import EnvName


class NumpyBuffer:
    def __init__(self, buffer_size: int, batch_size: int, game: EnvName = game_name) -> None:
        self.state_buffer = np.zeros((buffer_size,) + CONFIG[game]['state_shape'], dtype=np.float32)
        self.policy_buffer = np.zeros((buffer_size, CONFIG[game]['n_actions']), dtype=np.float32)
        self.value_buffer = np.zeros((buffer_size,), dtype=np.float32)
        self.batch_size = batch_size
        self.pointer = 0
        self.size = 0
        self.game = game
        self.buffer_size = buffer_size

    def add(self, data: tuple[NDArray, NDArray, float]) -> None:
        """先数据增强再添加
        :param data: (state,pi,q)
        """
        augmented_data = self.augment_data(data, self.game)
        for state, pi, q in augmented_data:
            self.append(state, pi, q)

    def append(self, state: NDArray, pi: NDArray, q: float) -> None:
        self.state_buffer[self.pointer] = state
        self.policy_buffer[self.pointer] = pi
        self.value_buffer[self.pointer] = q
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def augment_data(self, data: tuple[NDArray, NDArray, float], env_name: str) -> list[tuple[NDArray, NDArray, float]]:
        """通过旋转和翻转棋盘进行数据增强
            - ChineseChess 支持水平翻转
            - Gomoku 支持8种增强
        :param data: (state,pi,q)
        :param env_name:环境不同增强算法不同
         :return 增强后的列别[(state,pi,q)]"""
        state, pi, q = data
        augmented_samples = []
        if env_name == 'ChineseChess':
            from env.chess import ChineseChess  # 延迟导入
            indices = (0, 4, 5)
            for i in indices:
                if i == 5:
                    transformed_state = switch_side_ip(state.copy())
                else:
                    transformed_state = apply_symmetry(state.copy(), i)
                transformed_prob = mirror_action_policy(pi, i, ChineseChess.mirror_lr_actions,
                                                        ChineseChess.mirror_ud_actions)
                augmented_samples.append((transformed_state, transformed_prob, q))
            return augmented_samples

        if env_name == 'Gomoku':
            if state.shape[0] == state.shape[1]:
                indices = range(8)
            else:  # 非方形棋盘
                indices = (0, 2, 4, 5)
            for i in indices:
                transformed_state = apply_symmetry(state.copy(), i)
                transformed_prob = mirror_board_policy(pi.copy(), i, state.shape)
                augmented_samples.append((transformed_state, transformed_prob, q))
            return augmented_samples

        raise ValueError("Unknown env name")

    def get_batch(self) -> tuple[NDArray, NDArray, NDArray]:
        """随机采样一个 batch 进行训练
            :return (state,pi,q)
                - state:[B,C,H,W]
                - pi:[B,H]
                - q:[B]"""
        if self.size < self.batch_size:
            raise ValueError(f'No enough data! Current size is {self.size},required size is {self.batch_size}.')
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        states = self.state_buffer[indices].transpose(0, 3, 1, 2).copy()
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
        """加载buffer，支持按num增量添加，优化内存使用"""
        path = os.path.join(CONFIG['data_dir'], env_name, CONFIG['buffer_name'])
        if not os.path.exists(path):
            print(f"Buffer not found at '{path}', current length: {self.size}")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
            if data['game'] != self.game:
                raise ValueError('Game mismatch! Replay buffer loading Failed!')
            self.size = data['size']
            self.pointer = data['pointer']
            self.state_buffer[:self.size] = data['states'][:self.size]
            self.policy_buffer[:self.size] = data['pis'][:self.size]
            self.value_buffer[:self.size] = data['values'][:self.size]

            print(f"Replay buffer loaded successfully. Current length: {self.size}")

    def clear(self) -> None:
        self.size = 0
        self.pointer = 0

    def __len__(self):
        return self.size
