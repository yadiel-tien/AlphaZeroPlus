import errno
import inspect
import os
import threading
from typing import cast, Sequence

from numpy.typing import NDArray

import socket
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from utils.logger import get_logger
from utils.types import EnvName
from .client import require_infer_removal
from .engine import InferenceEngine
from .functions import recv, send
from utils.config import CONFIG
from .request import SocketRequest


class InferServer(InferenceEngine):
    def __init__(self, model_id: int, env_name: EnvName, max_listen_workers: int = 100):
        self.max_listen_workers = max_listen_workers
        self._listen_thread: threading.Thread | None = None
        self._listen_pool: ThreadPoolExecutor | None = None
        self.is_ready = False
        self._server_sock: socket.socket | None = None
        self.client_count = 0
        self.connection_lock = threading.Lock()
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
            self.logger.info(f'socket {self.socket_path} has been removed!')

    def _setup_socket(self) -> None:
        self.logger.info(f'starting socket {self.socket_path}')
        self._clean_socket()
        # 创建socket
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        self._server_sock.listen()
        self._server_sock.settimeout(1)  # 1s超时

        # 用线程池处理连接
        self._listen_pool = ThreadPoolExecutor(self.max_listen_workers)
        self.is_ready = True
        self.logger.info(f"[+] InferenceServer {self.name} listening at {self.socket_path}")

    def _listen_loop(self):
        """监听socket发来的请求"""
        self._setup_socket()
        while self.running:
            try:
                # accept会阻塞，无法检查running状态，设置超时继续
                conn, _ = self._server_sock.accept()
                with self.connection_lock:
                    self.client_count += 1
                self._listen_pool.submit(self.handle_client, conn)
            except socket.timeout:
                continue
            except OSError as e:
                if not self.running:
                    # 预期中的错误：服务器正在关闭
                    self.logger.debug(f"Server {self.name} socket closed during shutdown")
                    break
                elif e.errno == errno.EBADF:
                    # 坏的文件描述符：套接字已关闭
                    self.logger.info(f"Socket for {self.name} was closed unexpectedly")
                    break
                else:
                    # 其他非预期的OSError
                    self.logger.error(f"Unexpected OSError in {self.name}: {e}")
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
                    self.logger.info(f"[-] InferenceServer {self.name} received unsupported data: {data}")
            except socket.timeout:
                continue
            except ConnectionError:
                with self.connection_lock:
                    self.client_count -= 1
                    if self.client_count == 0:
                        require_infer_removal(self.name)
                break
        client_sock.close()

    def deliver_result(self, requests: list[SocketRequest], probs: Sequence[NDArray], values: Sequence[float]) -> None:
        """神经网络处理后的结果通过socket发回请求方"""
        for r, p, v in zip(requests, probs, values):
            send(r.sock, (p, v))

    def deliver_one(self, request: SocketRequest, prob: NDArray, value: float) -> None:
        send(request.sock, (prob, value))

    def shutdown(self) -> None:
        """清理资源"""
        logger = get_logger("debug")
        # 打印调用栈信息
        stack = inspect.stack()
        logger.info(f"{type(self)} {self.name} Shutdown was called from:")
        for frame_info in stack[1:5]:  # 显示前几层调用栈（避免显示shutdown自身）
            filename = frame_info.filename
            lineno = frame_info.lineno
            function = frame_info.function
            logger.info(f"  File '{filename}', line {lineno}, in {function}")
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
        self.logger.info(f'Cleaning up socket {self.socket_path} during shutdown')
        self._clean_socket()
