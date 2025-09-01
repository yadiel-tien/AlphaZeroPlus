import threading

import numpy as np
from numpy.typing import NDArray

from env.env import BaseEnv
from mcts.mcts import MCTS
from utils.types import EnvName
from .player import Player


class MCTSPlayer(Player):
    def __init__(self, env_name: EnvName, n_simulation=1000) -> None:
        super().__init__(env_name)
        self.mcts: MCTS | None = None
        self._thread: threading.Thread | None = None
        self._n_simulation = n_simulation
        self.description = 'MCTS'

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        print('思考中...')
        self._run_mcts(state, last_action)
        self.env_class.describe_move(state, self.pending_action)
        return self.pending_action

    def update(self, env: BaseEnv) -> None:
        if not self._thinking:
            # 新线程运行MCTS
            self.is_thinking = True
            self._thread = threading.Thread(target=self._run_mcts, args=(env.state.copy(), env.last_action),
                                            name='MCTS run')
            self._thread.start()

    def _run_mcts(self, state: np.ndarray, last_action: int) -> None:
        if self.mcts is None:
            self.mcts = MCTS(state)
        else:
            self.mcts.apply_opponent_action(state, last_action)
        # start = time.time()
        self.mcts.run(self._n_simulation)
        # print(f'{self._iteration} iteration took {time.time() - start:.2f} seconds')
        self.pending_action = self.mcts.choose_action()
        self._thinking = False

    def reset(self) -> None:
        super().reset()
        self.mcts = None
        self._thread = None
