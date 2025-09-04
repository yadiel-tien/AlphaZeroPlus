import os
import threading
from typing import cast
from numpy.typing import NDArray

import socket
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from utils.types import EnvName
from .client import require_infer_removal
from .engine import InferenceEngine
from .functions import recv, send
from utils.config import CONFIG
from .request import SocketRequest


class InferServer(InferenceEngine):
    def __init__(self, model_id: int, env_name: EnvName, max_listen_workers: int = 32):
        self.max_listen_workers = max_listen_workers
        self._listen_thread: threading.Thread | None = None
        self._listen_pool: ThreadPoolExecutor | None = None
        self._server_sock: socket.socket | None = None
        self.client_count = 0
        self.lock = threading.Lock()
        super().__init__(model_id, env_name)

    def start(self) -> None:
        """启动推理线程和监听进程"""
        super().start()
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

    @property
    def socket_path(self) -> str:
        """根据infer名称自动分配socket path"""
        return f'{CONFIG['socket_path_prefix']}{self.name}.sock'

    def _clean_socket(self) -> None:
        # 清理残留，否则可能会绑定失败
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

    def _setup_socket(self) -> None:
        self._clean_socket()
        # 创建socket
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        self._server_sock.listen()
        self._server_sock.settimeout(1)  # 1s超时

        # 用线程池处理连接
        self._listen_pool = ThreadPoolExecutor(self.max_listen_workers)
        print(f"[+] InferenceServer {self.name} listening at {self.socket_path}")

    def _listen_loop(self):
        """监听socket发来的请求"""
        self._setup_socket()
        while self.running:
            try:
                # accept会阻塞，无法检查running状态，设置超时继续
                conn, _ = self._server_sock.accept()
                with self.lock:
                    self.client_count += 1
                    print(f'New connection established, total {self.client_count} clients')
                self._listen_pool.submit(self.handle_client, conn)
            except socket.timeout:
                continue
            except OSError:
                print(f"[-] InferenceServer {self.name} listen loop was forced to shutdown!")
                break

    def handle_client(self, client_sock: socket.socket) -> None:
        """socket接收到state，通过队列发送给推理线程"""
        client_sock.settimeout(1)
        while self.running:
            try:
                data = recv(client_sock)
                if isinstance(data, np.ndarray):
                    self.request_queue.put(SocketRequest(cast(NDArray, data), client_sock))
                else:
                    print(f"[-] InferenceServer {self.name} received unsupported data: {data}")
            except socket.timeout:
                continue
            except ConnectionError:
                with self.lock:
                    self.client_count -= 1
                    if self.client_count == 0:
                        require_infer_removal(client_sock)
                break
        client_sock.close()

    def deliver_result(self, requests: list[SocketRequest], policies: NDArray[np.float32],
                       values: NDArray[np.float32]) -> None:
        """神经网络处理后的结果通过socket发回请求方"""
        for r, p, v in zip(requests, policies, values):
            send(r.sock, (p, v))

    def shutdown(self) -> None:
        """清理资源"""
        # 关闭推理进程
        super().shutdown()

        # 关闭服务socket
        if self._server_sock:
            self._server_sock.close()
            self._server_sock = None

        # 关闭监听线程
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=1)
            self._listen_thread = None

        # 关闭线程池
        if self._listen_pool:
            self._listen_pool.shutdown(wait=True)
            self._listen_pool = None

        # 清理socket目录
        self._clean_socket()
