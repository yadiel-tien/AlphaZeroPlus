import sys
import time

import numpy as np

from inference.client import send_request, send_via_queue
from inference.train_server import TrainServer
from utils.config import game_name
from utils.replay import NumpyBuffer
from utils.visualize import plot_gomoku_policy_heatmap

sys.path.append('/home/bigger/projects/five_in_a_row/')
import pickle

from env.gomoku import Gomoku
from env.chess import ChineseChess
from inference.engine import InferenceEngine as Infer
from player.human import Human
from player.ai_server import AIServer
from network.functions import read_latest_index


def main():
    latest = read_latest_index()

    env = ChineseChess()
    state, _ = env.reset()
    with AIServer(game_name, latest, ) as ai:
        action = ai.get_action(env.state, env.last_action, env.player_to_move)
    # # result = env.run(players)
    # # env.describe_result(result, players)
    buffer = NumpyBuffer(500000, 128)
    buffer.load()
    # batch = buffer.get_batch()
    # for i in range(32):
    #     state, pi, value = batch[0][i], batch[1][i], batch[2][i]
    #
    #     transposed=np.transpose(state, ( 1, 2,0))
    #     Gomoku.render_fn(transposed, 0)
    #     plot_gomoku_policy_heatmap(pi)
    #     print(value)


if __name__ == '__main__':
    main()
