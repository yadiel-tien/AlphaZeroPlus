from numpy.typing import NDArray

from utils.types import EnvName
from .player import Player


class Human(Player):
    def __init__(self, env_name: EnvName):
        super().__init__(env_name)
        self.selected_grid: tuple[int, int] | None = None

    @property
    def description(self) -> str:
        return 'Human'

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        return self.env_class.handle_human_input(state, last_action, player_to_move)

    def reset(self) -> None:
        super().reset()
        self.selected_grid = None
