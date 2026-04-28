import numpy as np
from numpy.typing import NDArray

from core.env.chess import ChineseChess


class NPZLoader:
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.states = data['states']
        self.last_actions = data['last_actions']
        self.step = int(data['steps'])

    def __len__(self):
        return len(self.states)

    def sample(self) -> tuple[NDArray, int, int]:
        """随机从数据中抽取，返回state，last_action,current steps"""
        idx = np.random.randint(len(self.states))
        state = self.states[idx].astype(np.int32)
        last_action: int = self.last_actions[idx]
        return state, last_action, self.step


if __name__ == "__main__":
    opening = NPZLoader("env/chess_step10.npz")
    env = ChineseChess()
    env.state, env.last_action, env.steps = opening.sample()
    env.player_to_move = env.steps % 2
    env.render()
    print(f'play_to_move: {env.player_to_move}')
    print(f'last action: {env.last_action}')
    print(f'move:{env.action2move(env.last_action)}')
    print(f'step:{env.steps}')
