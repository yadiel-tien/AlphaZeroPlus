from env.functions import get_class
from network.functions import read_latest_index
from player.human import Human
from player.ai_server import AIServer
from utils.config import game_name

if __name__ == "__main__":
    latest_model = read_latest_index()
    env = get_class(game_name)()
    with AIServer(game_name, latest_model, n_simulation=300) as ai, AIServer(game_name, 247,
                                                                              n_simulation=300) as ai2:
        result = env.run((ai, ai2))
        # result = env.run((Human(game_name),ai))
    env.render()
