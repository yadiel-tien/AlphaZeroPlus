from utils.types import EnvName
from .player import Player
from numpy.typing import NDArray
import random


class RandomPlayer(Player):
    def __init__(self, env_name: EnvName) -> None:
        super().__init__(env_name)

    @property
    def description(self) -> str:
        return 'Random Player'

    def update_state(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        self.pending_action = random.choice(self.env_class.get_valid_actions(state, player_to_move))
