import queue
import threading

from numpy.typing import NDArray
import socket

from .functions import send, recv, parse_socket_path
from utils.config import CONFIG
from utils.mirror import random_mirror_state_ip, mirror_action_policy, reverse_board_policy
from .request import QueueRequest
import time


def require_fit(iteration: int, n_exp: int) -> None:
    """请求train server更新训练模型"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(CONFIG['train_socket_path'])
        payload = {'command': 'fit',
                   'iteration': iteration,
                   'n_exp': n_exp}
        send(s, payload)


def require_train_server_shutdown() -> None:
    """通知train server 关闭"""
    _require_shutdown(CONFIG['train_socket_path'])


def require_hub_shutdown() -> None:
    """通知hub关闭"""
    _require_shutdown(CONFIG['hub_socket_path'])


def _require_shutdown(sock_path: str) -> None:
    """发送shutdown命令"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(sock_path)
        payload = {'command': 'shutdown'}
        send(s, payload)


def apply_for_socket_path(model_id: int, env_name: str) -> str:
    """在infer hub注册infer，注册后才会建立infer使用
    :return socket_path,用于建立到infer的socket连接"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(CONFIG['hub_socket_path'])
        payload = {'command': 'register', 'model_id': model_id, 'env_name': env_name}
        send(s, payload)
        socket_path = recv(s)
        return socket_path


def require_infer_removal(sock: socket.socket) -> None:
    """通知hub移除infer"""
    env_name, model_id = parse_socket_path(sock.getsockname())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(CONFIG['hub_socket_path'])
        payload = {'command': 'remove', 'env_name': env_name, 'model_id': model_id}
        send(s, payload)


def send_request(
        sock: socket.socket,
        state: NDArray,
        env_name: str,
        infer_queue: queue.Queue[QueueRequest] | None,
        is_self_play=False
) -> tuple[NDArray, float]:
    """根据是否提供queue来选择request的发送方式"""
    if infer_queue is None:
        return send_via_socket(sock, state, env_name, is_self_play)
    return send_via_queue(state, env_name, infer_queue, is_self_play)

#
# gpu_time = 0
# count = 0
# cpu_time = 0
# cpu_start = 0


def send_via_socket(sock: socket.socket, state: NDArray, env_name: str, is_self_play=False) -> tuple[NDArray, float]:
    """传入state，输出policy和value。is_self_play=True时对数据随机进行对称变换。
    适用于不同进程间进行请求和推理。设计用来3.14多线程selfplay发送请求到3.13推理进程"""
    state, symmetric_idx = preprocess_state(state, env_name, is_self_play)

    # 通过socket发送请求，再通过socket接收结果回传
    # global gpu_time, count, cpu_time, cpu_start

    send(sock, state)
    # t = time.time()
    # cpu_time += t - cpu_start
    #
    policy, value = recv(sock)
    # gpu_time += time.time() - t
    # cpu_start = time.time()
    # count += 1
    # if count % 500 == 0:
    #     print('500 requests GPU processing avg time:', gpu_time / count)
    #     print('500 requests CPU processing avg time:', cpu_time / count)
    #     gpu_time = 0
    #     cpu_time = 0
    policy = postprocess_policy(policy, symmetric_idx, env_name, state.shape)
    return policy, value


def send_via_queue(state: NDArray, env_name: str, q: queue.Queue, is_self_play=False) -> tuple[
    NDArray, float]:
    """传入state，输出policy和value。is_self_play=True时对数据随机进行对称变换。
    适用于通过多线程来请求和推理，请求和推理共用queue队列"""
    # 对state进行随机变换
    state, symmetric_idx = preprocess_state(state, env_name, is_self_play)

    # 通过queue发送请求
    event = threading.Event()
    request = QueueRequest(state, event=event)
    # 发给推理进程，等待结果
    q.put(request)
    event.wait()  # 阻塞等待推理线程处理完并设置 event

    policy = postprocess_policy(request.policy, symmetric_idx, env_name, state.shape)
    return policy, request.value


def preprocess_state(state: NDArray, env_name: str, is_self_play: bool) -> tuple[NDArray, int]:
    """selfplay时随机翻转state，并返回翻转id，否则不改变state，id=0"""
    symmetric_idx = 0
    if is_self_play:
        state, symmetric_idx = random_mirror_state_ip(state, env_name)
    return state, symmetric_idx


def postprocess_policy(policy: NDArray, symmetry_idx: int, env_name: str, board_shape: tuple) -> NDArray:
    """将policy翻转回去与原始的state相匹配"""
    if symmetry_idx == 0:
        return policy

    if env_name == 'ChineseChess':
        from env.chess import ChineseChess
        reverse_policy = mirror_action_policy(policy, symmetry_idx, ChineseChess.mirror_lr_actions,
                                              ChineseChess.mirror_ud_actions)
    elif env_name == 'Gomoku':
        reverse_policy = reverse_board_policy(policy, symmetry_idx, board_shape)
    else:
        raise RuntimeError(f'Unknown env_name: {env_name}')
    return reverse_policy
