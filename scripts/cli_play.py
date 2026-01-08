from env.functions import get_class
from network.functions import read_best_index
from player.human import Human
from player.ai_server import AIServer
from utils.config import game_name

if __name__ == "__main__":
    best_model = read_best_index()
    env = get_class(game_name)()
    with AIServer(game_name, best_model, n_simulation=500) as ai:
        # result = env.run((ai2, ai))
        result = env.run((Human(game_name), ai))
    env.render()
