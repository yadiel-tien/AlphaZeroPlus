import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from env.chess import ChineseChess
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
        self.n_workers = n_workers
        self.buffer = NumpyBuffer(500_000, 2048)
        self.buffer.load()
        self.env_class = get_class(game_name)

    def run(self, n_games: int) -> None:
        """训练入口
        :param n_games: 每轮的游戏数量"""
        iteration = read_latest_index()
        iteration = 1 if iteration == -1 else iteration + 1

        while iteration < settings['max_iters']:
            self.logger.info(f'Starting selfplay, iteration {iteration}')
            # 先保证buffer足够大
            while self.buffer.size < self.buffer.capacity * 0.4:
                self.self_play(iteration=iteration, n_games=50)
                self.logger.info(f'Collecting data.Current buffer size: {self.buffer.size}.')

            # 开始selfplay
            data_count = self.self_play(iteration=iteration, n_games=n_games)

            # 服务端进行模型训练，并保存参数，升级infer model
            require_fit(iteration, data_count)

            # 训练网络，保存网络
            iteration += 1

    def self_play(self, iteration: int, n_games=100) -> int:
        """自博弈收集数据
        :param iteration: 当前迭代轮次
        :param n_games: 每次自博弈进行的对局数量"""

        start = time.time()
        # 动态n_simulation,最大到1200
        n_simulation = 200 + iteration * 1000 // settings['max_iters']

        with  ThreadPoolExecutor(self.n_workers) as pool:
            futures = [pool.submit(self.self_play_worker, n_simulation) for _ in range(n_games)]
            data_count, win_count, lose_count, draw_count, truncate_count = 0, 0, 0, 0, 0
            for f in tqdm(as_completed(futures), total=n_games, desc='self playing'):
                samples, winner = f.result()
                win_count += winner == 0
                lose_count += winner == 1
                draw_count += winner == -1
                truncate_count += winner == 2
                data_count += len(samples)

                for sample in samples:
                    self.buffer.add(sample)

        self.buffer.save()
        # 总结
        duration = time.time() - start
        self.logger.info(
            f'selfplay {n_games}局游戏，每步模拟{n_simulation}次，收集到原始数据{data_count}条,\n'
            f'win rate:{win_count / n_games:.2%},lose rate:{lose_count / n_games:.2%},'
            f'draw rate:{draw_count / n_games :.2%},truncate rate:{truncate_count / n_games :.2%}。')
        self.logger.info(
            f'总用时{duration:.2f}秒, 平均步数{data_count / n_games:.2f}, 平均每条数据用时{(duration / data_count) if data_count else float('inf'):.4f}秒。'
        )
        return data_count

    def self_play_worker(self, n_simulation: int) -> tuple[list[
        tuple[NDArray, NDArray, float]], int]:
        """进行一局游戏，收集经验
        :return [(state,pi_move,q),...],winner.winner0,1代表获胜玩家，-1代表平局"""

        env = self.env_class()
        state, _ = env.reset()

        mcts = NeuronMCTS.make_selfplay_mcts(state=state,
                                             env_class=self.env_class,
                                             last_action=env.last_action,
                                             player_to_move=env.player_to_move)
        steps = 0
        samples = []
        while not env.terminated and not env.truncated:
            # Playout Cap Randomization (模拟次数随机化),丰富训练数据的多样性。
            n_simulation = random.randint(100, n_simulation)
            mcts.run(n_simulation)  # 模拟

            # 采集原始概率分布。象棋需要交换红黑双方位置对应的概率分布
            pi = mcts.get_pi(1.0)
            if env.player_to_move == 1 and isinstance(env, ChineseChess):
                pi_target = ChineseChess.switch_side_policy(pi)
            else:
                pi_target = pi
            # 象棋表示state和神经网络state不一样，需要转换。五子棋也进行了接口匹配
            state = env.convert_to_network(env.state, env.player_to_move)
            # q代表对上个玩家的回报，-q代表当前玩家的回报
            q = mcts.root.w / mcts.root.n
            samples.append((state, pi_target, -q, env.player_to_move))

            # 前期高温，后期低温。根据mcts模拟的概率分布进行落子
            # temperature = 1.0 if steps < settings['tao_switch_steps'] else 0.2
            # 临时测试使用固定温度1选择动作
            # temperature = 1.0
            # pi = mcts.get_pi(temperature)  # 获取mcts的概率分布pi
            action = np.random.choice(len(pi), p=pi)
            env.step(action)  # 执行落子
            mcts.apply_action(action)  # mcts也要根据action进行对应裁剪
            steps += 1

            # 避免大量无意义走棋，提前终止
            # 根据棋子数设定步数限制，如果有吃子，步数可以大些，吃子越多越大
            # if isinstance(env, ChineseChess):
            #     no_capture_count = round(float(state[0, 0, -1]) * 100)
            #     # piece_count = np.count_nonzero(env.state[:, :, 0] + 1)
            #     # step_limit = 100 - (piece_count - 9) * 3
            #     if no_capture_count > 30 or steps > 150:
            #         break

        mcts.shutdown()

        env.render()
        print(f'winner: {env.winner},steps: {steps}')

        # 平局或着截断数据大部分丢弃
        # if env.winner in (-1, 2) and random.random() < 0.5:
        #     return [], env.winner

        # 以当前玩家视角获取reward，胜1负-1平0
        for i in range(len(samples)):
            state, pi, q, p = samples[i]
            z = -1.0 if env.winner == 1 - p else 1.0 if env.winner == p else 0.0
            # q与z加权使用
            alpha = 0.5
            v = alpha * z + (1 - alpha) * q
            # print("DEBUG sample winner,p,z,q，v:", env.winner, p, z, q, v)

            samples[i] = state, pi, v

        return samples, env.winner

    def shutdown(self):
        require_train_server_shutdown()
