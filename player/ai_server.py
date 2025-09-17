from typing import Self

from numpy.typing import NDArray
from mcts.deepMcts import NeuronMCTS
from utils.types import EnvName
from .player import Player


class AIServer(Player):
    def __init__(self, env_name: EnvName, model_id: int, n_simulation=500, silent=False) -> None:
        super().__init__(env_name)
        self.model_id = model_id
        self._n_simulation = n_simulation
        self.mcts: NeuronMCTS | None = None
        self.silent = silent  # silent=True减少日志信息
        self.description = f'Server {self.model_id}'
        self.win_rate = 0.5

    def get_action(self, state: NDArray, last_action: int, player_to_move: int) -> int:
        """获取动作"""
        if not self.silent:
            print('思考中...')
        self.run_mcts(state, last_action, player_to_move)
        if not self.silent:
            self.env_class.describe_move(state, int(self.pending_action))
        return self.pending_action

    def run_mcts(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """运行mcts，根据模拟结果选择最优动作"""
        if self.mcts is None:
            self.mcts = NeuronMCTS.make_socket_mcts(
                state=state,
                env_class=self.env_class,
                last_action=last_action,
                player_to_move=player_to_move,
                model_id=self.model_id
            )
        else:
            self.mcts.apply_action(last_action)
        if not self.silent:
            print(f'before run: root width:{len(self.mcts.root.children)},depth:{self.mcts.root.depth}')
        self.mcts.run(self._n_simulation)
        if not self.silent:
            print(f'after run: root width:{len(self.mcts.root.children)},depth:{self.mcts.root.depth}')
        self.pending_action = self.mcts.choose_action()
        if not self.silent:
            print(f'after choose: root width:{len(self.mcts.root.children)},depth:{self.mcts.root.depth}')
            print(f'{self.description} win rate: {self.mcts.root.win_rate:.2%}')
        self.win_rate = self.mcts.root.win_rate
        self.is_thinking = False

    def reset(self) -> None:
        super().reset()
        self.shutdown()

    def shutdown(self) -> None:
        if self.mcts:
            self.mcts.shutdown()
            self.mcts = None

    def __del__(self) -> None:
        self.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
