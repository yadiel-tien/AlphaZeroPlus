import os
import pickle
import random

from tqdm import tqdm

from env.functions import get_class
from player.ai_server import AIServer
from utils.config import CONFIG
from utils.elo import Elo
from utils.logger import get_logger
from utils.types import EnvName
from concurrent.futures import ThreadPoolExecutor, as_completed


class Arena:
    def __init__(self, env_name: EnvName) -> None:
        self.rates: dict[int, Elo] = {}
        self.logger = get_logger('arena')
        self.env_name = env_name
        self.load()

    def run(self, n_games: int) -> None:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.vs_worker) for _ in range(n_games)]
            for future in tqdm(as_completed(futures), total=n_games):
                future.result()

    def vs_worker(self) -> None:
        index1, index2 = self.get_random_models()
        self.versus(index1, index2)

    def get_random_models(self) -> tuple[int, int]:
        """选择两个model id 以供比赛"""
        if len(self.rates) < 2:
            raise ValueError(f'No enough candidates,please check {self.list_path}.')
        model_indices = list(self.rates.keys())
        # 根据比赛次数多少随机，次数越多，选中概率越低
        weights = [1 / (self.rates[i].n_games_played + 1) for i in model_indices]
        chosen_index = random.choices(model_indices, weights=weights, k=1)[0]
        model_indices.remove(chosen_index)
        # 选择得分相近的对手，权重为1/abs(a-b)
        diff_weight = [1 / (
                abs(self.rates[idx].scores - self.rates[chosen_index].scores) + 1e-6
        ) for idx in model_indices]
        chosen_rival_index = random.choices(model_indices, weights=diff_weight, k=1)[0]
        return chosen_index, chosen_rival_index

    def versus(self, index1: int, index2: int) -> None:
        """index1与index2进行一局对弈，根据胜负结果调整ELO"""
        if index1 in self.rates and index2 in self.rates:
            self.logger.info(f'{index1} VS {index2}: Pre_game:{self.rates[index1]}  {self.rates[index2]}')
            env = get_class(self.env_name)()
            # 对弈
            with AIServer(self.env_name, index1) as p1, AIServer(self.env_name, index2) as p2:
                outcome = env.random_order_play((p1, p2), silent=True)
            # 更新score
            e1, e2 = self.rates[index1], self.rates[index2]
            if outcome == (1, 0):
                e1.defeat(e2)
            elif outcome == (0, 1):
                e2.defeat(e1)
            elif outcome == (0, 0):
                e1.draw(e2)
            self.save()
            self.logger.info(f'{index1} VS {index2}: Post_game:{self.rates[index1]}  {self.rates[index2]}')
        else:
            self.logger.info(f'{index1} or {index2} not found.')

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.list_path), exist_ok=True)
        with open(self.rates_path, "wb") as f:
            pickle.dump(self.rates, f, protocol=4)  # type: ignore
            self.logger.info(f'Arena scores have been saved at {self.rates_path}.')

    def load(self) -> None:
        # 读取rates
        if os.path.exists(self.rates_path):
            with open(self.rates_path, "rb") as f:
                self.rates = pickle.load(f)
            self.logger.info(f'loaded rates from {self.rates_path}')
        else:
            self.logger.info(f'Failed to load data from "{self.rates_path}",file not exist.')
        # 根据
        if os.path.exists(self.list_path):
            self.logger.info(f'Adjusting candidate list from {self.list_path}.')
            with open(self.list_path, "r") as f:
                lst_set = set(map(int, f.readline().strip().split(',')))
                for i in lst_set:
                    if i not in self.rates:
                        self.rates[i] = Elo(i)
                        self.logger.info(f'index:{i} joined arena.')
                for i in set(self.rates) - lst_set:
                    self.rates.pop(i)
                    self.logger.info(f'index:{i} was deleted from arena.')

    @property
    def rates_path(self) -> str:
        return os.path.join(CONFIG['rates_dir'], self.env_name, 'rates.pkl')

    @property
    def list_path(self) -> str:
        return os.path.join(CONFIG['rates_dir'], self.env_name, 'candidates.txt')

    def show_rank(self) -> None:
        """从高到低显示所有排名情况"""
        if not self.rates:
            self.logger.info('No candidate rates to show.')
            return
        for idx, (_, v) in enumerate(sorted(self.rates.items(), key=lambda x: x[1].scores, reverse=True)):
            self.logger.info(f'{idx:>3} {v}')
