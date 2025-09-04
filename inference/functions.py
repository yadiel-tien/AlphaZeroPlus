import os
import socket
import struct
import pickle
from typing import Any, cast

from utils.config import CONFIG
from utils.types import EnvName


def send(sock: socket.socket, obj: Any) -> None:
    """将对象序列化并带上长度头发送,格式为4字节长度信息，payload"""
    # 将数据转为字节
    # 如果不指定protocol会导致还原数据时卡住!!!
    payload = pickle.dumps(obj, protocol=4)
    # struct 在 Python 与二进制之间进行转换：数字 ↔ 字节流
    # !表示网络字节序（big endian），跨平台兼容 TCP/Unix _socket
    # I 表示一个无符号4字节整数（unsignedint，范围0~4, 294, 967, 295）
    # pack将长度数字转为字节
    length = struct.pack('!I', len(payload))
    sock.sendall(length + payload)


def recv(sock: socket.socket) -> Any:
    """接收payload，开头长度信息，获取完整payload返回"""
    # 4个字节的长度信息字节码
    raw_length = _recv_all(sock, 4)

    # unpack() 总是返回一个 tuple，哪怕只有一个字段，也得用 [0] 拿出来
    # 用unpack将字节转换为长度
    msg_length = struct.unpack('!I', raw_length)[0]
    # 获取payload
    data = _recv_all(sock, msg_length)
    return pickle.loads(data)


def _recv_all(sock: socket.socket, length: int) -> bytes:
    """阻塞接收直到收到指定字节数"""
    buf = b''
    while len(buf) < length:
        # sock.recv(length)有数据就会返回，最多为length长度，如果对方断开连接，则会返回b''
        chunk = sock.recv(length - len(buf))
        if not chunk:
            raise ConnectionError("_socket connection broken before receiving full payload!")
        buf += chunk
    return buf


def get_model_name(env_name: str, model_id: int) -> str:
    """根据env_name和model_id生成一致的model name"""
    return f'{env_name}_{model_id}'


def get_model_path(env_name: str, model_id: int) -> str:
    """根据env_name和model_id生成model的存储路径"""
    return os.path.join(CONFIG['data_dir'], env_name, f'model_{model_id}.pt')


def parse_socket_path(socket_path: str) -> tuple[EnvName, int]:
    """根据socket path解析出env和id"""
    name = socket_path.split('/')[-1].split('.')[0]
    env_name, model_index = name.split('_')
    env_name = cast(EnvName, env_name)
    return env_name, int(model_index)
