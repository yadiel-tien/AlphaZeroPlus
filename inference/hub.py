import datetime
import os
import socket
import threading
import time

from inference.functions import recv, send, get_model_name
from inference.infer_server import InferServer
from utils.config import CONFIG
from utils.logger import get_logger
from utils.types import EnvName


class ServerHub:
    """作为一个管理中心，负责推理infer的管理，包括新增，删除，不负责训练infer"""

    def __init__(self):
        self._socket = None
        # name:infer形式存储，可以让不同应用共享infer
        self.infers: dict[str, InferServer] = {}
        self._running = False
        self.logger = get_logger('hub')
        # 避免多线程同时操作infers造成数据错误
        self.lock = threading.Lock()

    def start(self) -> None:
        """启动管理服务"""
        if self._running:
            return
        self._running = True
        # 建立socket服务，地址在config中配置
        self._clean_socket()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(CONFIG['hub_socket_path'])
        self._socket.listen()
        # 设置超时，以便检查运行状态
        self._socket.settimeout(1)
        self.logger.info(f"Server hub started. Now is listening to {CONFIG['hub_socket_path']}.")
        # 启动状态显示
        threading.Thread(target=self.show_status, daemon=True).start()

        while self._running:
            try:
                conn, _ = self._socket.accept()
                threading.Thread(target=self.handle_connection, args=(conn,)).start()
            except socket.timeout:
                continue

    def handle_connection(self, conn: socket.socket) -> None:
        """响应客户请求，单独一个函数以便多线程处理"""
        try:
            with conn:
                data = recv(conn)
                if isinstance(data, dict):
                    # 根据不同的命令，进行不同的响应
                    if data['command'] == 'register':
                        send(conn, self.register(data['env_name'], data['model_id']))
                    elif data['command'] == 'remove':
                        name = get_model_name(data['env_name'], data['model_id'])
                        self.remove_infer(name)
                    elif data['command'] == 'shutdown':
                        self.shutdown()

        except ConnectionError as e:  # 对方断开
            self.logger.info(e)
        except socket.timeout:
            self.logger.info('Socket timed out')

    def register(self, env_name: EnvName, model_id: int) -> str:
        """注册infer，并返回连接该infer的socket地址"""
        name = get_model_name(env_name, model_id)
        with self.lock:
            if name not in self.infers:
                self.infers[name] = InferServer(model_id, env_name)
                self.infers[name].start()
        return self.infers[name].socket_path

    def remove_infer(self, model_name: str) -> None:
        """没有应用在使用infer，将其移除清理。检查使用情况在infer内部"""
        if model_name in self.infers:
            with self.lock:
                infer = self.infers.pop(model_name)
                infer.shutdown()
                self.logger.info(f'{model_name} has been removed!')
        else:
            self.logger.info(f'remove failed, model {model_name} is not registered!')

    def show_status(self):
        """定期显示连接数量"""
        while self._running:
            title = 'Connection Status'
            print(f'{title:-^70}')
            with self.lock:
                for name, infer in self.infers.items():
                    print(f'{name}: {infer.client_count} clients')
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"):-^70}')
            time.sleep(30)

    @staticmethod
    def _clean_socket() -> None:
        """使用前后都需要清理socket文件"""
        if os.path.exists(CONFIG['hub_socket_path']):
            os.remove(CONFIG['hub_socket_path'])

    def shutdown(self):
        """清理资源"""
        self._running = False
        # 关闭socket
        if self._socket:
            self._socket.close()
            self._socket = None
        self._clean_socket()
        # 清理infer
        with self.lock:
            for infer in self.infers.values():
                infer.shutdown()
            self.infers.clear()
        self.logger.info('Server hub has been shut down!')
