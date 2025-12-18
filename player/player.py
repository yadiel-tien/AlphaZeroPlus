from numpy.typing import NDArray
from env.functions import get_class
from utils.types import EnvName


class Player:
    def __init__(self, env_name: EnvName):
        self.pending_action: int = -1
        self.is_thinking: bool = False
        self.env_class = get_class(env_name)

    @property
    def description(self) -> str:
        return "Player"

    def update(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """在UI界面下无阻塞更新，会被频繁调用"""
        pass

    def update_state(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """同步当前数据给玩家，玩家在此基础上做选择"""
        pass

    def get_action(self) -> int:
        """必须要在执行update_state后再调用"""
        return self.pending_action

    def reset(self) -> None:
        self.pending_action = -1
        self.is_thinking = False
