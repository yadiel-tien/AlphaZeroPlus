from core.utils.functions import register_sigint

from services.inference.hub import ServerHub


def main():
    hub = ServerHub()

    register_sigint(hub.shutdown)

    hub.start()


if __name__ == '__main__':
    main()
