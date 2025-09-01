import sys

sys.path.append('/home/bigger/projects/five_in_a_row/')
import pickle

from env.gomoku import Gomoku
from env.chess import ChineseChess
from inference.engine import InferenceEngine as Infer
from player.human import Human
from player.ai_server import AIServer
from network.functions import read_latest_index


def test_infer():
    latest_index = read_latest_index()
    latest_infer = Infer(latest_index)
    zero_infer = Infer(-1)
    env = Gomoku()

    players = [Human(env.__class__.__name__),
               AIServer(latest_infer, env.__class__.__name__, n_simulation=1000)]
    result = env.run(players, False)
    env.describe_result(result, players, False)


def read_test_data():
    with open('/home/bigger/projects/five_in_a_row/data/data123.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data)


if __name__ == '__main__':
    test_infer()
