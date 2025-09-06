import sys

sys.path.append('/home/bigger/projects/five_in_a_row/')
from utils.functions import register_sigint
from train.selfplay import SelfPlayManager


def main():
    manager = SelfPlayManager(36)

    register_sigint(manager.shutdown)

    manager.run(iteration=200, n_games=200)

    manager.shutdown()


if __name__ == '__main__':
    main()
