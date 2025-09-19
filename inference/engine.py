import queue
import threading
import time
from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from utils.logger import get_logger
from utils.types import EnvName
from .functions import get_checkpoint_path, get_model_name
from .request import QueueRequest, SocketRequest
from utils.config import CONFIG
from network.network import Net


class InferenceEngine:
    def __init__(self, model_index: int, env_name: EnvName, training: bool = False):
        # 推理model
        model_path = get_checkpoint_path(env_name, model_index)
        self.eval_model, success, = Net.load_from_checkpoint(model_path, eval_model=True)
        self.model_index = model_index if success else -1
        self.env_name = env_name
        # 负责单个request的发送接收
        self.request_queue: queue.Queue[QueueRequest | SocketRequest] = queue.Queue()
        self._request_thread: threading.Thread | None = None
        # 负责多个request组成batch发送推理和结果分发
        self.infer_queue: queue.Queue[list[QueueRequest | SocketRequest]] = queue.Queue()
        self._infer_thread: threading.Thread | None = None
        self.running = False
        self.model_lock = threading.Lock()
        self.training = training
        self.logger = get_logger('inference')
        if self.training:
            # 置换表
            self.transposition_table = OrderedDict()
            self.tt_max_size = 200_000
            self.hit = 0
            self.total_request = 0
            self.clear_flag = False

    @property
    def name(self):
        return get_model_name(self.env_name, self.model_index)

    def make_chinese_chess_state_key(self, state: NDArray) -> bytes:
        """供置换表使用"""
        arr = (state[:, :, :14] > 0.5).astype(np.uint8)  # 保证0/1
        return arr.tobytes()

    def start(self) -> None:
        """启动推理线程"""
        if not self.running:
            self.running = True
            self._request_thread = threading.Thread(target=self._collect_loop, daemon=True)
            self._request_thread.start()
            self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._infer_thread.start()

    def _collect_loop(self):
        """从request queue收集数据，尽可能多的收集数据打包"""
        while self.running:
            batch_size = 1
            threshold = 8
            max_size = 20
            n_pending = self.infer_queue.qsize()
            delay = 8e-4 + 6e-4 * n_pending  # 根据queue排队情况，动态调整
            phase = 'ramp up'
            requests = []

            while len(requests) < min(batch_size, max_size):
                # 思路参考tcp拥塞控制
                try:
                    request = self.request_queue.get(timeout=delay)
                    requests.append(request)
                    if phase == 'ramp up':
                        if batch_size < threshold:
                            batch_size *= 2
                        else:
                            batch_size += 1
                            phase = 'steady increase'
                    elif phase == 'steady increase':
                        batch_size += 1

                except queue.Empty:
                    threshold = max(1, batch_size // 2)
                    batch_size = threshold
                    phase = 'ramp up'
            if requests:
                self.infer_queue.put(requests)

    def _inference_loop(self):
        """推理loop，持续不断的接收state，打包，发GPU推理，返回结果"""
        # start_time = time.time()
        while self.running:
            start = time.time()
            # 收到清空信号重置置换表
            if self.training and self.clear_flag:
                self.clear_flag = False
                self.hit = 0
                self.total_request = 0
                self.transposition_table.clear()

            # 获取推理列表
            try:
                requests = self.infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # 避免get阻塞不能结束循环

            # 检查缓存
            if self.training:
                miss_list = []
                for request in requests:
                    self.total_request += 1
                    key = self.make_chinese_chess_state_key(request.state)
                    if key in self.transposition_table:  # 命中
                        policy, value = self.transposition_table[key]
                        self.deliver_one(request, policy, value)
                        # LRU
                        self.transposition_table.move_to_end(key)
                        self.hit += 1
                    else:
                        miss_list.append(request)
            else:
                miss_list = requests
            if not miss_list:
                continue

            # 处理batch，转为tensor
            batch = [np.transpose(request.state, axes=[2, 0, 1]) for request in miss_list]
            batch_tensor = torch.from_numpy(np.stack(batch)).to(CONFIG['device'], dtype=torch.float32)

            # 交模型推理，取回结果
            with torch.no_grad():
                with torch.amp.autocast('cuda'):  # 混合精度
                    with self.model_lock:
                        logits, values = self.eval_model(batch_tensor)
            probs = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
            values = values.float().cpu().numpy()
            self.deliver_result(miss_list, probs, values)

            # 更新缓存
            if self.training:
                for request, prob, value in zip(miss_list, probs, values):
                    key = self.make_chinese_chess_state_key(request.state)
                    self.transposition_table[key] = (prob, value)
                # LRU清理旧缓存
                while len(self.transposition_table) > self.tt_max_size:
                    self.transposition_table.popitem(last=False)

                msg = f'{len(miss_list):>2} requests inference took {time.time() - start:.10f} seconds.Pending:{self.infer_queue.qsize():2}.'
                msg += f' hit rate: {self.hit / (self.total_request + 1) :.2%}, total:{self.total_request}.'
                print(msg, end='\r')
                # if time.time() - start_time > 10:
                #     self.logger.debug(msg)
                #     start_time = time.time()

    def deliver_result(self, requests: list[QueueRequest], probs: Sequence[NDArray], values: Sequence[float]) -> None:
        # 结果交给请求方，通知request继续
        for r, p, v in zip(requests, probs, values):
            r.policy = p
            r.value = v
            r.event.set()

    def deliver_one(self, request: QueueRequest, prob: NDArray, value: float) -> None:
        request.policy = prob
        request.value = value
        request.event.set()

    def shutdown(self) -> None:
        """清理资源，关闭推理线程"""
        self.running = False
        if self._request_thread and self._request_thread.is_alive():
            self._request_thread.join(timeout=1)
            self._request_thread = None
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1)
            self._infer_thread = None
        if self.training:
            self.transposition_table.clear()

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
