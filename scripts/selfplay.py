from utils.functions import register_sigint
from train.selfplay import SelfPlayManager


def main():
    manager = SelfPlayManager(n_workers=100)

    register_sigint(manager.shutdown)

    try:
        manager.run(100)
        # manager.evaluation(455,50)
        manager.shutdown()
    except ConnectionError or FileNotFoundError:
        print("Server has been shut down,selfplay stopped!")


if __name__ == '__main__':
    main()
    # require_fit(415,5000)
