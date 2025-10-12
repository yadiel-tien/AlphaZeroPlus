import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from env.chess import ChineseChess
from env.functions import get_class
from inference.functions import get_checkpoint_path
from utils.logger import get_logger
from utils.state_buffer import StateBuffer
from utils.replay import ReplayBuffer
from utils.config import game_name, settings
from mcts.deepMcts import NeuronMCTS
from inference.client import require_fit, require_train_server_shutdown, require_update_eval_model, \
    require_restore_fit_model
from network.functions import read_latest_index, save_best_index, read_best_index
import random


class SelfPlayManager:
    def __init__(self, n_workers: int):
        self.logger = get_logger('selfplay')
        self.n_workers = n_workers
        self.buffer = ReplayBuffer(500_000, 2048)
        self.buffer.load()
        self.env_class = get_class(game_name)
        self.best_index = read_best_index()
        self.pool = ThreadPoolExecutor(self.n_workers)
        self.midgame_buffer = StateBuffer(2000, 'midgame')
        self.opening_buffer = StateBuffer(2000, 'opening')
        self.midgame_buffer.load()
        self.opening_buffer.load()
        self.debug_logger = get_logger('debug')

    def run_bi_game(self, n_games: int) -> None:
        """交替训练两个游戏，使网络具有掌握两种游戏的泛化能力"""
        pass

    def run(self, n_games: int) -> None:
        """训练入口
        :param n_games: 每轮的游戏数量"""
        iteration = read_latest_index()
        iteration = 1 if iteration == -1 else iteration + 1

        while iteration < settings['max_iters']:
            self.logger.info(f'Starting selfplay, iteration: {iteration}, best index: {self.best_index}.')
            # 先保证buffer足够大
            while self.buffer.size < self.buffer.capacity * 0.4:
                self.self_play(iteration=iteration, n_games=50)
                self.logger.info(f'Collecting data.Current buffer size: {self.buffer.size}.')

            # 开始selfplay。新数据会进行积累，直到新模型产生。
            n_data = self.self_play(iteration=iteration, n_games=n_games)

            # 服务端进行模型训练，并保存参数，升级infer model
            done = require_fit(iteration, n_data)
            print(done)

            # 以防模型还没创建好
            path = get_checkpoint_path(game_name, iteration=iteration)
            while not os.path.exists(path):
                self.logger.info(f'Checkpoint {path} does not exist. Retrying.')
                time.sleep(1)
            # 评估模型，按结果更新模型
            self.evaluation(iteration, n_games=50)

            # 训练网络，保存网络
            iteration += 1

    def self_play(self, iteration: int, n_games=100) -> int:
        """自博弈收集数据
        :param iteration: 当前迭代轮次
        :param n_games: 每次自博弈进行的对局数量"""

        start = time.time()
        # 动态n_simulation,最大到1200
        # n_simulation = 200 + iteration * 600 // settings['max_iters']
        n_simulation = 800
        stop_signal = threading.Event()
        futures = [self.pool.submit(self.self_play_worker, n_simulation, stop_signal) for _ in
                   range(int(n_games * 1.2))]
        data_count, win_count, lose_count, draw_count, truncate_count, completed = 0, 0, 0, 0, 0, 0
        with tqdm(total=n_games, desc='Self-play') as pbar:
            for future in as_completed(futures):
                try:
                    samples, winner = future.result()
                except Exception as e:
                    self.logger.error(f'Exception occurred: {e}')
                    stop_signal.set()
                    break
                pbar.update(1)

                completed += 1
                win_count += winner == 0
                lose_count += winner == 1
                draw_count += winner == -1
                truncate_count += winner == 2
                data_count += len(samples)

                for sample in samples:
                    self.buffer.add(sample)
                if completed >= n_games:
                    stop_signal.set()
                    break

        self.buffer.save()
        self.midgame_buffer.save()
        self.opening_buffer.save()
        # 总结
        duration = time.time() - start
        self.logger.info(
            f'selfplay {completed}局游戏，每步模拟{n_simulation}次，收集到原始数据{data_count}条,\n'
            f'win rate:{win_count / completed:.2%},lose rate:{lose_count / completed:.2%},'
            f'draw rate:{draw_count / completed :.2%},truncate rate:{truncate_count / completed :.2%}。')
        self.logger.info(
            f'总用时{duration:.2f}秒, 平均步数{data_count / completed:.2f}, 平均每条数据用时{(duration / data_count) if data_count else float('inf'):.4f}秒。'
        )
        return data_count

    def self_play_worker(self, n_simulation: int, stop_signal: threading.Event) -> tuple[list[
        tuple[NDArray, NDArray, float]], int]:
        """进行一局游戏，收集经验
        :return [(state,pi_move,q),...],winner.winner0,1代表获胜玩家，-1代表平局"""

        env = self.env_class()
        env.reset()
        # 25%概率从残局开始
        start_from_beginning = True
        if random.random() < 0.5 and len(self.midgame_buffer) > 50:
            start_from_beginning = False
            env.state = self.midgame_buffer.sample()

        mcts = NeuronMCTS.make_selfplay_mcts(state=env.state,
                                             env_class=self.env_class,
                                             last_action=env.last_action,
                                             player_to_move=env.player_to_move)
        steps = 0
        samples = []
        while not env.terminated and not env.truncated:
            if steps == 8 and start_from_beginning and (0.4 < mcts.root.win_rate < 0.6):
                self.opening_buffer.append(env.state)
            if steps == 50 and start_from_beginning and (0.4 < mcts.root.win_rate < 0.6):
                self.midgame_buffer.append(env.state)

            mcts.run(n_simulation)  # 模拟

            # 采集原始概率分布。象棋需要交换红黑双方位置对应的概率分布

            pi_target = mcts.get_pi(1.0)
            if env.player_to_move == 1 and isinstance(env, ChineseChess):
                pi_target = ChineseChess.switch_side_policy(pi_target)
            # 象棋表示state和神经网络state不一样，需要转换。五子棋也进行了接口匹配
            state = env.convert_to_network(env.state, env.player_to_move)
            # q代表对上个玩家的回报，-q代表当前玩家的回报
            q = mcts.root.w / mcts.root.n
            samples.append((state, pi_target, -q, env.player_to_move))

            # 前期高温，后期低温。根据mcts模拟的概率分布进行落子
            if start_from_beginning and steps < settings['tao_switch_steps']:
                temperature = 1.0
            else:
                temperature = 0.1
            pi = mcts.get_pi(temperature)  # 获取mcts的概率分布pi
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

            # 收到停止信号，截断游戏
            if stop_signal.is_set():
                env.truncated = True

        mcts.shutdown()

        # 截断的不收集数据
        if env.truncated:
            return [], env.winner
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

    def evaluation(self, iteration: int, n_games: int) -> None:
        """评估模型，对战就模型胜率超过55%才更新模型"""
        start_time = time.time()
        # 额外提交20%任务，总数达到后停止，避免个别长时间等待
        stop_signal = threading.Event()
        futures = [self.pool.submit(self.evaluation_worker, iteration, stop_signal) for _ in
                   range(int(n_games * 1.2))]
        win_count, lose_count, draw_count, win_rate, total_steps = 0, 0, 0, 0.0, 0
        # 进度条
        with tqdm(total=n_games, desc=f'{iteration} VS {self.best_index}') as pbar:
            for future in as_completed(futures):
                winner, steps = future.result()
                pbar.update(1)
                total_steps += steps
                win_count += winner == 0
                lose_count += winner == 1
                draw_count += winner == -1
                completed = win_count + lose_count + draw_count
                win_rate = (win_count + draw_count / 2) / completed
                pbar.set_postfix({'win_rate': f'{win_rate:.2%}'})  # 进度条后面添加当前胜率

                if completed >= n_games:
                    stop_signal.set()
                    break

        self.logger.info(
            f'Model {iteration} VS {self.best_index}，胜:{win_count},负:{lose_count},平:{draw_count}, 胜率{win_rate:.2%}。')
        if win_rate >= 0.55:  # 通过测试
            self.best_index = iteration
            save_best_index(iteration)
            require_update_eval_model(iteration)
            self.logger.info(
                f'更新最佳模型为{iteration}，平均步数{total_steps // n_games}步，评估用时{time.time() - start_time:.2f}秒.')
        else:
            self.logger.info(
                f'最佳模型仍旧为{self.best_index}，平均步数{total_steps // n_games}步， 评估用时{time.time() - start_time:.2f}秒.')
            require_restore_fit_model(best_index=self.best_index, iteration=iteration)

    def evaluation_worker(self, iteration: int, stop_signal: threading.Event) -> tuple[int, int]:
        """iteration对战最佳模型，随机先手顺序。
        :return 0新模型胜，1老模型胜，-1平"""
        env = self.env_class()
        # 随机前8步开局，增加随机性
        if len(self.opening_buffer) > 50:
            env.state = self.opening_buffer.sample()
        # 随机先后手
        model_list = [iteration, self.best_index] if random.random() < 0.5 else [self.best_index, iteration]
        competitors = [NeuronMCTS.make_socket_mcts(
            env_class=self.env_class,
            state=env.state,
            last_action=env.last_action,
            player_to_move=env.player_to_move,
            model_id=index
        ) for index in model_list]
        # 随机模拟测试
        n_simulation = 300
        steps = 0
        while not env.terminated and not env.truncated:
            mcts = competitors[env.player_to_move]
            mcts.run(n_simulation)
            action = int(np.argmax(mcts.root.child_n))
            env.step(action)
            # 双方都要对应剪枝
            for mcts in competitors:
                mcts.apply_action(action)
            if stop_signal.is_set():
                env.truncated = True
            steps += 1

        for mcts in competitors:
            mcts.shutdown()
        if env.winner in (-1, 2):  # 和棋或被提前终止
            return env.winner, steps

        winner = model_list[env.winner]

        env.render()
        print(f'winner: {env.winner},tester win:{winner == iteration},steps: {steps}')

        if winner == iteration:
            return 0, steps
        else:
            return 1, steps

    def shutdown(self):
        self.pool.shutdown()
        require_train_server_shutdown()
