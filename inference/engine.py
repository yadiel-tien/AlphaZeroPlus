import queue
import threading
import time
from lru import LRU
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
    def __init__(self, model_index: int, env_name: EnvName, verbose=False):
        self.verbose = verbose
        # 推理model
        model_path = get_checkpoint_path(env_name, model_index)
        self.infer_model, success, = Net.load_from_checkpoint(model_path, eval_model=True)
        self.model_index = model_index if success else -1
        self.env_name = env_name
        # 负责单个request的发送接收
        self.request_queue: queue.Queue[QueueRequest | SocketRequest] = queue.Queue()
        self._collector_thread: threading.Thread | None = None
        # 负责多个request组成batch转为tensor
        self.preprocess_queue: queue.Queue[list[QueueRequest | SocketRequest]] = queue.Queue()
        self._preprocess_thread: threading.Thread | None = None
        # 负责推理
        self.infer_queue: queue.Queue[tuple[list[QueueRequest | SocketRequest], torch.Tensor]] = queue.Queue()
        self._infer_thread: threading.Thread | None = None
        # 负责推理结果的处理
        self.result_queue: queue.Queue[tuple[list[QueueRequest | SocketRequest], NDArray, NDArray]] = queue.Queue()
        self._result_thread: threading.Thread | None = None

        self.running = False
        self.model_lock = threading.Lock()
        self.logger = get_logger('inference')
        # 置换表
        self.transposition_table = LRU(200_000)
        self.cache_hits = 0
        self.finished_requests = 0
        self.start_time = time.time()
        self.total_requests = 0
        self.clear_flag = False

    @property
    def name(self):
        return get_model_name(self.env_name, self.model_index)

    def make_state_key(self, state: NDArray) -> bytes:
        """供置换表使用"""
        if self.env_name == "ChineseChess":
            arr = (state[:, :, :14] > 0.5).astype(np.uint8)  # 保证0/1
        else:
            arr = state.astype(np.uint8)
        return arr.tobytes()

    def start(self) -> None:
        """启动推理线程"""
        if not self.running:
            self.running = True
            self._collector_thread = threading.Thread(target=self._collect_loop, daemon=True)
            self._collector_thread.start()
            self._preprocess_thread = threading.Thread(target=self._pre_infer_loop, daemon=True)
            self._preprocess_thread.start()
            self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._infer_thread.start()
            self._result_thread = threading.Thread(target=self._post_infer_loop, daemon=True)
            self._result_thread.start()

    def _collect_loop(self):
        """从request queue收集数据，尽可能多的收集数据打包"""
        self.start_time = time.time()
        while self.running:
            batch_size = 1
            threshold = 8
            max_size = 32
            n_pending = self.infer_queue.qsize()
            delay = 1e-4 + 1e-3 * n_pending  # 根据queue排队情况，动态调整
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
                    if not self.running:
                        break
            if requests:
                self.preprocess_queue.put(requests)

    def _pre_infer_loop(self):
        """做推理前准备工作，将batch_queue接收到的数据查缓存，打包，转tensor"""
        while self.running:
            # 收到清空信号重置置换表
            if self.clear_flag:
                self.clear_flag = False
                self.cache_hits = 0
                self.total_requests = 0
                self.transposition_table.clear()
                self.start_time = time.time()
                self.finished_requests = 0

            # 获取推理列表
            try:
                requests = self.preprocess_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 检查缓存
            miss_list: list[QueueRequest | SocketRequest] = []
            for request in requests:
                self.total_requests += 1
                key = self.make_state_key(request.state)
                if key in self.transposition_table:  # 命中
                    policy, value = self.transposition_table[key]
                    self.deliver_one(request, policy, value)
                    self.cache_hits += 1
                    self.finished_requests += 1
                else:
                    miss_list.append(request)
            if not miss_list:
                continue

            # 处理batch，转为tensor
            batch = [np.transpose(request.state, axes=[2, 0, 1]) for request in miss_list]
            batch_tensor = torch.from_numpy(np.stack(batch)).pin_memory()
            self.infer_queue.put((miss_list, batch_tensor))

    def _inference_loop(self):
        """推理loop，持续不断的接收state，打包，发GPU推理，返回结果"""
        while self.running:
            try:
                miss_list, tensor = self.infer_queue.get(timeout=0.5)
                batch_tensor = tensor.to(CONFIG['device'], dtype=torch.float32, non_blocking=True)
                # 交模型推理，取回结果
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):  # 混合精度
                        with self.model_lock:
                            logits, values = self.infer_model(batch_tensor)
                probs = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
                values = values.float().cpu().numpy()
                self.result_queue.put((miss_list, probs, values))
            except queue.Empty:
                continue

    def _post_infer_loop(self):
        while self.running:
            try:
                miss_list, probs, values = self.result_queue.get(timeout=0.5)
                self.deliver_result(miss_list, probs, values)

                # 更新缓存
                for request, prob, value in zip(miss_list, probs, values):
                    key = self.make_state_key(request.state)
                    self.transposition_table[key] = (prob.copy(), float(value))
                self.finished_requests += len(miss_list)
                # if self.verbose:
                msg = f'Batch size: {len(miss_list):>2}.Pending:{self.infer_queue.qsize():2}.'
                msg += f' hit rate: {self.cache_hits / (self.total_requests + 1) :.2%}, total:{self.total_requests}.'
                msg += f' cost:{(time.time() - self.start_time) * 1000 / self.finished_requests:.6f}sec per 1000 requests.'
                print(msg, end='\r')

            except queue.Empty:
                continue

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
        if self.infer_model:
            del self.infer_model
            self.infer_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 清空队列
        for q in [self.request_queue, self.preprocess_queue, self.infer_queue, self.result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        for t in [self._collector_thread, self._preprocess_thread, self._infer_thread, self._result_thread]:
            if t:
                t.join()

        self._collector_thread = self._preprocess_thread = self._infer_thread = self._result_thread = None

        self.transposition_table.clear()

        # 尝试强制释放 numpy 内存回 OS,否则置换表的内存无法回收
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception as e:
            print("malloc_trim not available:", e)

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
