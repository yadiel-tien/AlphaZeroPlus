import sys

sys.path.append('/home/bigger/projects/five_in_a_row')
import os
import pickle

from utils.config import CONFIG
from utils.replay import NumpyBuffer

if __name__ == '__main__':
    buffer = NumpyBuffer(50000, 128)
    path = os.path.join(CONFIG['data_dir'], buffer.game, CONFIG['buffer_name'])
    if not os.path.exists(path):
        print(f"Buffer not found at '{path}', current length: {buffer.size}")

    with open(path, "rb") as f:
        data = pickle.load(f)
        for state, pi, q in data:
            buffer.append(state, pi, q)
    buffer.save()
