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
        pass

    def get_action(self) -> int:
        pass

    def reset(self) -> None:
        self.pending_action = -1
        self.is_thinking = False
