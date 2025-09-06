import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from env.functions import get_class
from utils.logger import get_logger
from utils.replay import NumpyBuffer
from utils.config import game_name, settings
from mcts.deepMcts import NeuronMCTS
from inference.client import require_fit, require_train_server_shutdown
from network.functions import read_latest_index
import random


class SelfPlayManager:
    def __init__(self, n_workers: int):
        self.logger = get_logger('selfplay')
        self.latest_model_index = read_latest_index()
        self.n_workers = n_workers
        self.buffer = NumpyBuffer(500_000, 2048)
        self.buffer.load()
        self.env_class = get_class(game_name)

    def run(self, iteration: int, n_games: int) -> None:
        """训练入口
        :param iteration: 主循环，eval_model index以此产生
        :param n_games: 每轮的游戏数量"""
        if self.latest_model_index == -1:
            self.latest_model_index = 0
        for i in range(iteration):
            self.logger.info(
                f'iteration {i + 1}/{iteration},latest: {self.latest_model_index}')
            # self_play
            self.logger.info(f'selfplay iteration {i + 1}/{iteration},latest: {self.latest_model_index}')
            while self.buffer.size < self.buffer.capacity * 0.8:
                self.self_play(iteration=self.latest_model_index, n_games=50)
                self.logger.info(f'Collecting data.Current buffer size: {self.buffer.size}.')
            data_count = self.self_play(iteration=self.latest_model_index, n_games=n_games)

            # 训练网络，保存网络
            self.latest_model_index += 1

            # 服务端进行模型训练，并保存参数，升级infer model
            require_fit(self.latest_model_index, data_count)

    def self_play(self, iteration: int, n_games=100) -> int:
        """自博弈收集数据
        :param iteration: 当前迭代轮次
        :param n_games: 每次自博弈进行的对局数量"""

        start = time.time()
        with  ThreadPoolExecutor(self.n_workers) as pool:
            futures = [pool.submit(self.self_play_worker, iteration) for _ in range(n_games)]
            game_count, data_count, draw_count, truncate_count = 0, 0, 0, 0
            for f in tqdm(as_completed(futures), total=n_games, desc='self playing'):
                experiences, winner = f.result()
                game_count += 1
                draw_count += winner == -1
                truncate_count += winner == 2
                data_count += len(experiences)

                for data in experiences:
                    self.buffer.add(data)

        self.buffer.save()
        # 总结
        duration = time.time() - start
        self.logger.info(
            f'selfplay {n_games}局游戏，收集到原始数据{data_count}条,draw rate:{draw_count / game_count :.2%},truncate rate:{truncate_count / game_count :.2%}。')
        self.logger.info(
            f'总用时{duration:.2f}秒, 平均步数{data_count / n_games:.2f}, 平均每条数据用时{(duration / data_count) if data_count else float('inf'):.4f}秒。'
        )
        return data_count

    def self_play_worker(self, iteration: int) -> tuple[list[
        tuple[NDArray, NDArray, float]], int]:
        """进行一局游戏，收集经验
        :return [(state,pi_move,q),...],winner.winner0,1代表获胜玩家，-1代表平局"""

        env = self.env_class()
        state, _ = env.reset()

        mcts = NeuronMCTS.make_selfplay_mcts(state=state,
                                             env_class=self.env_class,
                                             last_action=env.last_action,
                                             player_to_move=env.player_to_move)
        step = 0
        experiences = []
        while not env.terminated and not env.truncated:
            # 动态n_simulation
            n_simulation = 200 + min(iteration / 200, 2) * 300
            # n_simulation = 300
            mcts.run(int(n_simulation))  # 模拟

            pi_target = mcts.get_pi(1.0)
            temperature = 1.0 if step < settings['tao_switch_steps'] else 0.2
            pi = mcts.get_pi(temperature)  # 获取mcts的概率分布pi

            # 象棋表示state和神经网络state不一样，需要转换。五子棋也进行了接口匹配
            state = env.convert_to_network(env.state, env.player_to_move)
            # 前20轮用z，后面改为q
            experiences.append((state, pi_target, env.player_to_move))

            action = np.random.choice(len(pi), p=pi)
            env.step(action)  # 执行落子
            mcts.apply_action(action)  # mcts也要根据action进行对应裁剪
            step += 1

            # 避免大量无意义走棋，提前终止
            # 根据棋子数设定步数限制，如果有吃子，步数可以大些，吃子越多越大
            if self.env_class.__name__ == 'ChineseChess':
                no_capture_count = round(float(state[0, 0, -1]) * 100)
                # piece_count = np.count_nonzero(env.state[:, :, 0] + 1)
                # step_limit = 100 - (piece_count - 9) * 3
                if no_capture_count > 30 or step > 150:
                    break

        mcts.shutdown()

        env.render()
        print(f'winner: {env.winner},steps: {step}')

        # 平局或着截断数据大部分丢弃
        if env.winner in (-1, 2) and random.random() < 0.5:
            return [], env.winner

        # 以当前玩家视角获取reward，胜1负-1平0
        for i in range(len(experiences)):
            state, pi, p = experiences[i]
            z = -1.0 if env.winner == 1 - p else 1.0 if env.winner == p else 0.0
            experiences[i] = (state, pi, z)

        return experiences, env.winner

    def shutdown(self):
        require_train_server_shutdown()
