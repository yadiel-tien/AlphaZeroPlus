from utils.functions import register_sigint
from train.selfplay import SelfPlayManager


def main():
    manager = SelfPlayManager(n_workers=40)

    register_sigint(manager.shutdown)

    try:
        manager.run(n_games=200)
        manager.shutdown()
    except ConnectionError or FileNotFoundError:
        print("Server has been shut down,selfplay stopped!")




if __name__ == '__main__':
    main()
