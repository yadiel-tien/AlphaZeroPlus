import argparse
from services.inference.play_api import PlayApiServer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZenZero Play Server")
    parser.add_argument('--standalone', type=str2bool, default=True, help="Automatically start local Inference Hub")
    parser.add_argument('--port', type=int, default=5000, help="Listen port")
    args = parser.parse_args()

    server = PlayApiServer(standalone_hub=args.standalone)
    server.run(port=args.port)
