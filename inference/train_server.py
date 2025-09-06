from typing import cast
import time
import torch
from numpy.typing import NDArray

import socket
import numpy as np

from network.network import Net
from utils.replay import NumpyBuffer
from utils.types import EnvName
from .infer_server import InferServer
from .functions import recv, get_model_path
from utils.config import CONFIG
from utils.logger import get_logger
from .request import SocketRequest


class TrainServer(InferServer):
    def __init__(self, model_id: int, env_name: EnvName, max_listen_workers: int = 100):
        super().__init__(model_id, env_name, max_listen_workers)
        self.logger = get_logger('fit')
        self.fit_model, _ = Net.make_model(self.model_index, env_name)
        self.fit_model.to(CONFIG['device'])
        self.optimizer = torch.optim.Adam(self.fit_model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.buffer = NumpyBuffer(500_000, 2048)

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
                print(f"[-] InferenceServer {self.name} listen loop was forced to shutdown: {e}")
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
                        self.fit(n_exp=data['n_exp'], iteration=data['iteration'])
                    elif data['command'] == 'shutdown':  # 关闭整个train server
                        self.shutdown()
                    else:
                        print(f'[-] Received unsupported command: {data["command"]}')
                else:
                    print(f'[-] Received unsupported data: {data}')
            except socket.timeout:
                continue
            except ConnectionError:  # 对方断开了连接，结束循环
                break
        client_sock.close()

    def update_from_index(self, index: int):
        """根据编号index，读取存储的模型，直接修改eval_model指针，避免多线程数据冲突"""
        model, self.model_index = Net.make_model(index, self.env_name)
        model.to(CONFIG['device']).eval()
        self.eval_model = model

    @property
    def socket_name(self) -> str:
        return self._server_sock.getsockname()

    def fit(self, n_exp: int, iteration: int) -> None:
        """从buffer中获取数据，训练神经网络"""

        start = time.time()
        # 加载最新buffer
        self.buffer.load()
        epochs = n_exp * 20 // self.buffer.batch_size
        for epoch in range(epochs):
            # 批量数据获取
            states, pis, zs = self.buffer.get_batch()
            states = torch.from_numpy(states).float().to(CONFIG['device'])
            pis = torch.from_numpy(pis).float().to(CONFIG['device'])
            zs = torch.from_numpy(zs).float().to(CONFIG['device'])
            print("\033[K", end='')  # 清空之前的尾行推理信息
            self.logger.info(
                f"real z mean: {torch.mean(zs).item():>7.4f},"
                f" std:{torch.std(zs).item():>7.4f},"
                f" q[:3]: {','.join(f'{i.item():>8.4f}' for i in zs[:3])}"
            )
            # 模型前向推理
            policy_logits, values = self.fit_model(states)
            self.logger.info(
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
            # 这些用于检查value的参数情况，判断有没有出现梯度消失
            # for name, param in self.fit_model.value.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.mean()}", end=', ')
            #     else:
            #         print(f"{name}:None", end=', ')
            # print()
            torch.nn.utils.clip_grad_norm_(self.fit_model.parameters(), max_norm=1.0)  # 将梯度范数裁剪到1.0
            self.optimizer.step()  # 更新参数

            self.logger.info(
                f"Epoch {epoch + 1}:\n "
                # f'w_policy={w_policy:.4f}, w_value={w_value:.4f}'
                f" loss={loss.item():.4f}, policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
            )
        # 保存新模型参数
        model_path = get_model_path(self.env_name, iteration)
        torch.save(self.fit_model.state_dict(), model_path)
        # 更新推理模型
        with self.model_lock:
            self.eval_model.load_state_dict(self.fit_model.state_dict())
        self.model_index = iteration
        duration = time.time() - start
        self.logger.info(f"iteration{iteration}:{epochs}轮训练完成，共用时{duration:.2f}秒。")

    def shutdown(self) -> None:
        super().shutdown()
