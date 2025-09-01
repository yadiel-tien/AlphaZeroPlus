import re

from gymnasium import spaces

from numpy.typing import NDArray

from utils.config import CONFIG
from utils.types import GomokuMove, GameResult
import numpy as np

from .env import BaseEnv

settings = CONFIG['Gomoku']


class Gomoku(BaseEnv):
    shape: tuple[int, int, int] = 15, 15, 2
    n_actions = shape[0] * shape[1]

    def __init__(self, rows: int = 15, columns: int = 15):
        super().__init__()
        self.shape: tuple[int, int, int] = rows, columns, 2
        self.action_space = spaces.Discrete(rows * columns)
        self.n_actions = rows * columns
        self.observation_space = spaces.Box(0, 1, shape=self.shape, dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None) -> tuple[NDArray[np.int_], dict]:
        """
        重置游戏, 返回当前棋盘状态
        array=[Xt, Yt]
        Xt 1代表有当前玩家棋子，0代表无当前玩家棋子，Yt 1代表有对手玩家棋子，0代表无对手玩家棋子
        :return state,info
        """
        self.state = np.zeros(self.shape, dtype=np.float32)
        self.reset_status()
        return self.state, {}

    @classmethod
    def convert_to_network(cls, state: NDArray, current_player: int) -> NDArray:
        return state.copy()

    @classmethod
    def get_valid_actions(cls, state: NDArray, player_to_move: int) -> NDArray[np.int_]:
        state = state[:, :, 0] + state[:, :, 1]
        return np.flatnonzero(state == 0)

    @classmethod
    def virtual_step(cls, state: NDArray[np.float32], action: int) -> NDArray[np.float32]:
        """只改变state，不计算输赢和奖励"""
        new_state = np.copy(state)
        row, col = cls.action2move(action)
        # 执行落子
        new_state[row, col, 0] = 1

        # 更改棋盘和当前玩家
        new_state[:, :, [0, 1]] = new_state[:, :, [1, 0]]
        return new_state

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        """
        执行落子
        :param action: 动作编号（棋盘上的位置）
        :return: observation（新的state）, reward（奖励）, terminated（是否结束）,truncated(是否因时间限制中断）, info（额外信息）
        """
        # 执行落子
        self.state, reward, terminated, _, _ = self.step_fn(self.state, action, self.player_to_move)
        if terminated:
            winner = self.player_to_move if reward == 1 else -1
            self.set_winner(winner)

        # 更改玩家
        self.player_to_move = 1 - self.player_to_move
        self.last_action = action

        # 胜利奖励1，平局或未结束奖励0，隐含失败后对手奖励1，通过min-max自己奖励-1
        reward = 0.0
        if self.winner == 1 - self.player_to_move:  # 落子导致胜利
            reward = 1.0

        return self.state, reward, terminated, self.truncated, {}

    @classmethod
    def check_winner(cls, state: NDArray, player_just_moved: int, action_just_executed: int) -> GameResult:
        """:return 1胜，0平，-1负, 2未分胜负"""
        if action_just_executed < 0:  # 刚开局的情况
            return GameResult.ONGOING
        # 和棋
        if cls._is_draw(state):
            return GameResult.DRAW
        # 检查刚下的棋子是否连成5子，只会返回1
        if cls.get_win_stones(state, action_just_executed):
            return GameResult.WIN
        return GameResult.ONGOING

    @classmethod
    def move2action(cls, move: GomokuMove) -> int:
        """从 (row, col) 坐标获取动作编号"""
        row, col = move
        return row * cls.shape[1] + col

    @classmethod
    def action2move(cls, action: int) -> GomokuMove:
        """从动作编号获取坐标 (row, col)"""
        return divmod(int(action), cls.shape[1])

    @classmethod
    def describe_move(cls, state: NDArray, action: int) -> None:
        """无UI对弈时，打印描述行棋的说明"""
        row, col = cls.action2move(action)
        print(f'选择落子：({row + 1},{col + 1})')

    @staticmethod
    def _is_draw(state: NDArray) -> bool:
        return np.all(np.logical_or(state[:, :, 0], state[:, :, 1]))

    @classmethod
    def get_win_stones(cls, state: NDArray, action_just_executed: int) -> list[tuple[int, int]]:
        """落子后检查获胜的情况，获胜返回连成5子的棋子位置，未获胜返回空列表[]"""
        h, w, _ = cls.shape
        h0, w0 = cls.action2move(action_just_executed)
        for dh, dw in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
            stones = [(h0, w0)]
            for direction in (-1, 1):
                for step in range(1, 5):
                    i, j = h0 + step * dh * direction, w0 + step * dw * direction
                    if 0 <= i < h and 0 <= j < w and state[i, j, 1]:
                        stones.append((i, j))
                        if len(stones) == 5:
                            return stones
                    else:
                        break
        return []

    def render(self) -> None:
        self.render_fn(self.state, self.player_to_move)

    @classmethod
    def render_fn(cls, state: NDArray, player_to_move: int) -> None:
        """打印棋盘"""
        board_str = '\n'
        col_indices = [str(i + 1) for i in range(cls.shape[1])]
        head = '  '
        for idx in col_indices:
            head += f'{idx:>3}'
        board_str += head + '\n'
        for i, row in enumerate(state):
            board_str += f'{i + 1:>2}  ' + '  '.join(
                ['X' if cell[player_to_move] else 'O' if cell[1 - player_to_move] else '.'
                 for cell in row]) + '\n'
            # 用红色显示玩家 1 的棋子 (1)
        board_str = re.sub(r'\bX\b', '\033[31mX\033[0m', board_str)
        # 用蓝色显示玩家 2 的棋子 (2)
        board_str = re.sub(r'\bO\b', '\033[34mO\033[0m', board_str)
        print(board_str)

    @classmethod
    def handle_human_input(cls, state: NDArray, last_action: int, player_to_move: int) -> int:
        cls.render_fn(state, player_to_move)
        valid_actions = cls.get_valid_actions(state, player_to_move)
        while True:
            while True:
                txt = input('输入落子位置坐标，示例"1,2"代表第1行第2列:')
                txt = txt.replace('，', ',')
                pos = txt.split(',')
                if len(pos) == 2 and type(pos) is list and pos[0].isdigit() and pos[1].isdigit():
                    break
                else:
                    print("输入格式有误，请输入行列编号，逗号隔开。")

            move = (int(pos[0]) - 1, int(pos[1]) - 1)
            action = cls.move2action(move)
            if action in valid_actions:
                break
            else:
                print("输入位置不合法，请重新输入！")
        return action
