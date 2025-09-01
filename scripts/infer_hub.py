import sys

sys.path.append('/home/bigger/projects/five_in_a_row')
from utils.functions import register_sigint

from inference.hub import ServerHub


def main():
    hub = ServerHub()

    register_sigint(hub.shutdown)

    hub.start()


if __name__ == '__main__':
    main()
