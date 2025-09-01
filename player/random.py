from utils.types import EnvName
from .player import Player
from numpy.typing import NDArray
import random


class RandomPlayer(Player):
    def __init__(self, env_name: EnvName) -> None:
        super().__init__(env_name)

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        return random.choice(self.env_class.get_valid_actions(state, player_to_move))
