import queue
import threading
import time

import numpy as np
import torch
from numpy.typing import NDArray

from utils.types import EnvName
from .functions import get_model_name
from .request import QueueRequest, SocketRequest
from utils.config import CONFIG
from network.network import Net


class InferenceEngine:
    def __init__(self, model_index: int, env_name: EnvName):
        # 推理model
        self.eval_model, self.model_index = Net.make_model(model_index, env_name)
        self.eval_model.to(CONFIG['device']).eval()
        self.env_name = env_name
        self.name = get_model_name(env_name, model_index)
        # 负责单个request的发送接收
        self.request_queue: queue.Queue[QueueRequest | SocketRequest] = queue.Queue()
        self._request_thread: threading.Thread | None = None
        # 负责多个request组成batch发送推理和结果分发
        self.infer_queue: queue.Queue[list[QueueRequest | SocketRequest]] = queue.Queue()
        self._infer_thread: threading.Thread | None = None
        self.running = False
        self.model_lock = threading.Lock()

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
            max_size = 12
            n_pending = self.infer_queue.qsize()
            delay = 5e-4 + 3e-4 * n_pending  # 根据queue排队情况，动态调整
            phase = 'ramp up'
            requests = []

            while self.running:
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

                    if len(requests) >= min(batch_size, max_size):
                        break
                except queue.Empty:
                    threshold = max(1, batch_size // 2)
                    batch_size = threshold
                    if len(requests) >= batch_size:
                        break
                    else:
                        phase = 'ramp up'
            if requests:
                self.infer_queue.put(requests)

    def _inference_loop(self):
        """推理loop，持续不断的接收state，打包，发GPU推理，返回结果"""
        while self.running:
            start = time.time()
            try:
                requests = self.infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # 避免get阻塞不能结束循环
            # 处理batch，转为tensor
            batch = [np.transpose(request.state, axes=[2, 0, 1]) for request in requests]
            batch_tensor = torch.from_numpy(np.stack(batch)).to(CONFIG['device'], dtype=torch.float32)

            # 交模型推理，取回结果
            with torch.no_grad():
                with torch.amp.autocast('cuda'):  # 混合精度
                    with self.model_lock:
                        logits, values = self.eval_model(batch_tensor)
            probs = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
            values = values.float().cpu().numpy()

            self.deliver_result(requests, probs, values)
            print(
                f'{len(requests):>2} requests inference took {time.time() - start:.10f} seconds.Pending:{self.infer_queue.qsize():2}.',
                end='\r')

    def deliver_result(self, requests: list[QueueRequest], probs: NDArray[np.float32],
                       values: NDArray[np.float32]) -> None:
        # 结果交给请求方，通知request继续
        for r, p, v in zip(requests, probs, values):
            r.policy = p
            r.value = v
            r.event.set()

    def shutdown(self) -> None:
        """清理资源，关闭推理线程"""
        self.running = False
        if self._request_thread and self._request_thread.is_alive():
            self._request_thread.join(timeout=1)
            self._request_thread = None
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1)
            self._infer_thread = None

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
