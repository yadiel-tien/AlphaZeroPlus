import json
import threading
import time
from typing import Any

import numpy as np
import requests

from env.env import BaseEnv
from utils.types import EnvName
from .player import Player
from utils.config import CONFIG


class AIClient(Player):
    def __init__(self, model_idx: int, env_name: EnvName) -> None:
        super().__init__(env_name)
        self.model_idx = model_idx
        self.pid = ''
        self.description = f'AI({model_idx})'
        self.time_stamp = 0
        self.session = requests.Session()
        self.request_setup()
        self.alive = True
        self.win_rate = 0.5
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def update(self, env: BaseEnv) -> None:
        """负责启动推理线程"""
        if not self.is_thinking:
            # 新线程运行MCTS
            self.is_thinking = True
            threading.Thread(target=self.request_move, args=(env.state, env.last_action, env.player_to_move),
                             daemon=True).start()

    def request_move(self, state: np.ndarray, last_action: int, player_to_move: int) -> None:
        """给server发请求，获取action"""
        url = CONFIG['base_url'] + 'make_move'
        payload = {
            'array': state.tolist(),
            'action': last_action,
            'pid': self.pid,
            'player_to_move': player_to_move
        }
        response = self.post_request(url, payload)

        self.pending_action = response.get('action')
        self.win_rate = response.get('win_rate')
        self.is_thinking = False

    def request_reset(self) -> None:
        """告知server重置"""
        url = CONFIG['base_url'] + 'reset'
        payload = {'pid': self.pid}
        self.post_request(url, payload)

    def _heartbeat_loop(self):
        while self.alive:
            self.send_heartbeat()
            time.sleep(5)

    def send_heartbeat(self) -> None:
        """通过定期发送心跳告知服务端存活，以再服务端保留资源"""
        self.time_stamp = time.time()
        url = CONFIG['base_url'] + 'heartbeat'
        payload = {'pid': self.pid}
        response = self.post_request(url, payload)
        self.win_rate = response.get('win_rate')

    def request_setup(self) -> None:
        """告知server创建推理引擎"""
        url = CONFIG['base_url'] + 'setup'
        payload = {'model_id': self.model_idx, 'env_class': self.env_class.__name__}
        response = self.post_request(url, payload)
        self.pid = response.get('pid')

    def post_request(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """发送请求的基础方法，获取反馈，处理错误"""
        response = None
        try:
            headers = {'content-type': 'application/json'}
            response = self.session.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                timeout=40  # 添加超时设置
            )
            response.raise_for_status()  # 自动处理4xx/5xx错误

            return response.json()

        except requests.exceptions.HTTPError as http_err:
            error_msg = f'HTTP错误 ({response.status_code if response else "Unknown"}): '
            try:
                error_data = response.json()
                error_msg += str(error_data.get('error', error_data))
            except ValueError:
                error_msg += response.text or str(http_err)
            print(error_msg)
            raise  # 重新抛出异常

        except json.JSONDecodeError as json_err:
            error_msg = f'响应不是有效的JSON: {str(json_err)}'
            print(error_msg)
            raise requests.exceptions.RequestException(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f'请求失败: {str(e)}'
            print(error_msg)
            raise  # 重新抛出异常

    def reset(self) -> None:
        self.request_reset()
        super().reset()
        self.win_rate = 0.5

    def shutdown(self) -> None:
        self.alive = False
