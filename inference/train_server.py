import os
from datetime import datetime
from typing import cast
import time
import torch
from numpy.typing import NDArray

import socket
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from network.functions import read_latest_index
from network.network import Net
from utils.replay import ReplayBuffer
from utils.types import EnvName
from .infer_server import InferServer
from .functions import recv, get_checkpoint_path, send
from utils.config import CONFIG, settings
from utils.logger import get_logger
from .request import SocketRequest


class TrainServer(InferServer):
    def __init__(self, model_id: int, env_name: EnvName, max_listen_workers: int = 100):
        super().__init__(model_id, env_name, max_listen_workers, verbose=True)
        self.fit_logger = get_logger('fit')
        # 模型
        self.fit_model = Net(self.infer_model.config, eval_model=False)
        # 优化器和学习率调解器
        self.optimizer = torch.optim.Adam(self.fit_model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                    T_max=settings['max_iters'], eta_min=1e-5)
        # 训练步数计数器
        self.total_steps_trained = 0
        # buffer
        self.buffer = ReplayBuffer(500_000, 2048)
        # 加载最新存档
        self.load_checkpoint(read_latest_index(self.env_name))
        # 可视化记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f'runs/{self.env_name}_{timestamp}')

    def load_checkpoint(self, iteration: int) -> None:
        """读取并加载存档"""
        path = get_checkpoint_path(self.env_name, iteration)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=CONFIG['device'])
            self.fit_model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.total_steps_trained = checkpoint['total_steps_trained']
            self.fit_logger.info(
                f'Load checkpoint successfully. Iteration: {iteration}, Step: {self.total_steps_trained}.')
        else:
            self.fit_logger.info(f'Checkpoint {iteration} not found.Starting from raw model.')

    def save_checkpoint(self, iteration: int) -> None:
        """保存存档"""
        # 创建checkpoint
        checkpoint = {
            'total_steps_trained': self.total_steps_trained,
            'model': self.fit_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.fit_model.config
        }

        # 保存新模型参数
        path = get_checkpoint_path(self.env_name, iteration)
        torch.save(checkpoint, path)

    @property
    def socket_path(self) -> str:
        return CONFIG['train_socket_path']

    def _listen_loop(self) -> None:
        """监听socket发来的请求"""
        self._setup_socket()
        while self.running:
            try:
                # accept会阻塞，无法检查running状态，设置超时继续
                conn, _ = self._server_sock.accept()
                self._listen_pool.submit(self.handle_client, conn)
            except socket.timeout:
                continue
            except OSError as e:
                self.fit_logger.info(f"[-] InferenceServer {self.name} listen loop was forced to shutdown: {e}")
                break
            except Exception as e:
                if not self.running:
                    break
                print(e)

    def handle_client(self, client_sock: socket.socket) -> None:
        """socket接收到state，通过队列发送给推理线程"""
        client_sock.settimeout(1)
        while self.running:
            try:
                data = recv(client_sock)
                if isinstance(data, np.ndarray):  # 接收到state，放入推理队列
                    self.request_queue.put(SocketRequest(cast(NDArray, data), client_sock))
                elif isinstance(data, dict) and 'command' in data:
                    if data['command'] == 'fit':  # 训练模型
                        self.fit(client_sock, n_collected_samples=data['n_exp'], iteration=data['iteration'])
                    elif data['command'] == 'shutdown':  # 关闭整个train server
                        self.shutdown()
                    elif data['command'] == 'update_eval_model':  # 训练有效，更新推理模型
                        with self.model_lock:
                            self.infer_model.load_state_dict(self.fit_model.state_dict())
                            self.model_index = data['iteration']
                        # 清空缓存
                        self.clear_flag = True
                        self.fit_logger.info(f'Evaluation model updated to {data['iteration']}.')
                    elif data['command'] == 'restore_fit_model':  # 训练失败，回滚学习模型
                        # self.load_checkpoint(data['best_index'])
                        path = get_checkpoint_path(self.env_name, data['iteration'])
                        if os.path.exists(path):
                            os.remove(path)
                        self.fit_logger.info(
                            f'Model{data['iteration']} does not pass evaluation, checkpoint was deleted.')
                    else:
                        self.fit_logger.info(f'[-] Received unsupported command: {data["command"]}')
                else:
                    self.fit_logger.info(f'[-] Received unsupported data: {data}')
            except socket.timeout:
                continue
            except ConnectionError:  # 对方断开了连接，结束循环
                break
        client_sock.close()

    @property
    def socket_name(self) -> str:
        return self._server_sock.getsockname()

    def fit(self, sock: socket, n_collected_samples: int, iteration: int) -> None:
        """从buffer中获取数据，训练神经网络"""
        start = time.time()
        # 加载最新buffer
        self.buffer.load()
        n_training_steps = n_collected_samples * settings['augment_times'] * CONFIG[
            'training_steps_per_sample'] // self.buffer.batch_size
        for step in range(n_training_steps):
            # 批量数据获取
            states, pis, zs = self.buffer.get_batch()
            states = torch.from_numpy(states).float().to(CONFIG['device'])
            pis = torch.from_numpy(pis).float().to(CONFIG['device'])
            zs = torch.from_numpy(zs).float().to(CONFIG['device'])
            print("\033[K", end='')  # 清空之前的尾行推理信息
            self.fit_logger.info(
                f"real z mean: {torch.mean(zs).item():>7.4f},"
                f" std:{torch.std(zs).item():>7.4f},"
                f" q[:3]: {','.join(f'{i.item():>8.4f}' for i in zs[:3])}"
            )
            # 模型前向推理
            policy_logits, values = self.fit_model(states)
            self.fit_logger.info(
                f"pred v mean: {torch.mean(values).item():>7.4f},"
                f" std:{torch.std(values).item():>7.4f},"
                f" v[:3]: {','.join(f'{i.item():>8.4f}' for i in values[:3])}"
            )

            # 交叉熵损失，使policy的结果趋近mcts模拟出来的pi，[B,H*W]->scalar
            policy_loss = - torch.sum(pis * torch.log_softmax(policy_logits, dim=1), dim=1).mean()
            # 均方差损失，使value的结果趋近与mcts模拟出来的z，[B]。
            value_loss = torch.nn.functional.mse_loss(values, zs) * 5

            # 用总的损失进行反向梯度更新
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()  # 清空旧梯度
            loss.backward()  # 反向传播

            # 检查梯度，debug用
            # print("--- Value Head Gradients ---")
            # for name, param in self.fit_model.value.named_parameters():
            #     if param.grad is not None:
            #         # 打印梯度的平均绝对值，比均值更能反映梯度大小
            #         print(f"{name}: {param.grad.abs().mean().item():.6f}", end=', ')
            #     else:
            #         print(f"{name}:None", end=', ')
            # print("\n--------------------------")

            torch.nn.utils.clip_grad_norm_(self.fit_model.parameters(), max_norm=1.0)  # 将梯度范数裁剪到1.0
            self.optimizer.step()  # 更新参数

            # 记录数据到tensorboard
            self.writer.add_scalar('Loss/total', loss.item(), self.total_steps_trained)
            self.writer.add_scalar('Loss/policy', policy_loss.item(), self.total_steps_trained)
            self.writer.add_scalar('Loss/value', value_loss.item(), self.total_steps_trained)

            # 更新计步器
            self.total_steps_trained += 1

            self.fit_logger.info(
                f"Step {step + 1}:\n "
                f" loss={loss.item():.4f}, policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
            )

        # 更新学习率调解器
        self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], self.total_steps_trained)
        self.scheduler.step()
        # 保存存档
        self.save_checkpoint(iteration)

        duration = time.time() - start
        self.fit_logger.info(f"iteration{iteration}:{n_training_steps}轮训练完成，共用时{duration:.2f}秒。")
        # 通知服务端学习完成，可以进行评估
        send(sock, 'Fit done!')

    def shutdown(self) -> None:
        if self.writer:
            self.writer.close()
        super().shutdown()
