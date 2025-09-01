import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray

from env.chess import ChineseChess
from env.gomoku import Gomoku


def plot_chess_policy_heatmap(pi: NDArray[np.float32]):
    """显示象棋概率的热力图"""
    board_shape = (10, 9)
    heatmap_from = np.zeros(board_shape, dtype=np.float32)
    heatmap_to = np.zeros(board_shape, dtype=np.float32)

    for action, prob in enumerate(pi):  # type:int,float
        r, c, to_r, to_c = ChineseChess.action2move(action)
        heatmap_from[r, c] += prob
        heatmap_to[to_r, to_c] += prob

    # 画图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(heatmap_from, ax=axs[0], annot=True, fmt=".2f", cmap="Blues", cbar=False)
    axs[0].set_title("Action From Probability")

    sns.heatmap(heatmap_to, ax=axs[1], annot=True, fmt=".2f", cmap="Reds", cbar=False)
    axs[1].set_title("Action To Probability")

    for ax in axs:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    plt.tight_layout()
    plt.show()


def plot_gomoku_policy_heatmap(pi: NDArray[np.float32], board_shape=(15, 15)) -> None:
    heatmap = np.zeros(board_shape, dtype=np.float32)
    for action, prob in enumerate(pi):
        r, c = Gomoku.action2move(action)
        heatmap[r, c] += prob

    # 画图
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(heatmap, ax=ax, annot=True, fmt=".3f", cmap="Blues", cbar=False)
    ax.set_title("Action Probability")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    # 设置坐标标签从 1 开始
    ax.set_xticks(np.arange(board_shape[1]) + 0.5)
    ax.set_yticks(np.arange(board_shape[0]) + 0.5)
    ax.set_xticklabels([str(i) for i in range(1, board_shape[1] + 1)])
    ax.set_yticklabels([str(i) for i in range(1, board_shape[0] + 1)])

    plt.tight_layout()
    plt.show()
