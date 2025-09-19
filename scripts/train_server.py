from utils.functions import register_sigint
from utils.config import game_name
import time
from inference.train_server import TrainServer
from network.functions import read_best_index


def main():
    # 启动推理服务
    model_idx = read_best_index()
    with TrainServer(model_idx, game_name) as server:
        register_sigint(server.shutdown)
        while server.running:
            time.sleep(1)  # 阻塞主线程，避免退出


if __name__ == '__main__':
    main()
