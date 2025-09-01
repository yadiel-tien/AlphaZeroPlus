import os.path
import pickle

from utils.config import game_name, CONFIG
from utils.types import EnvName


class EMA:
    def __init__(self, alpha=0.8, w1=1, w2=1) -> None:
        self.alpha = alpha
        self.w1 = w1
        self.w2 = w2

    def update(self, x1: float, x2: float) -> None:
        self.w1 = self.w1 * (1 - self.alpha) + self.alpha * x2
        self.w2 = self.w2 * (1 - self.alpha) + self.alpha * x1

    def get_weights(self) -> tuple[float, float]:
        total = self.w1 + self.w2 + 1e-8
        w1 = min(max(self.w1 / total, 0.05), 0.95)
        return w1, 1 - w1

    def save(self, env_name: EnvName = game_name) -> None:
        folder = os.path.join(CONFIG['data_dir'], env_name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, CONFIG['ema_name'])
        with open(path, 'wb') as f:
            pickle.dump((self.w1, self.w2), f)

    def load(self, env_name: EnvName = game_name) -> None:
        path = os.path.join(CONFIG['data_dir'], env_name, CONFIG['ema_name'])
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.w1, self.w2 = pickle.load(f)
        else:
            print(f'Ema load failed, "{path}" not found!')
