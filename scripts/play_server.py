import sys

sys.path.append('/home/bigger/projects/five_in_a_row/')
import threading
import uuid
import time
from functools import wraps
from typing import Callable, Any

import traceback

import numpy as np
from flask import Flask, request, jsonify

from player.ai_server import AIServer

app = Flask(__name__)
AIes: dict[str, AIServer] = {}
clients: dict[str, float] = {}


def require_json(f: Callable) -> Callable:
    """将获取data，检查data的内容抽取成装饰漆"""

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        return f(data, *args, **kwargs)

    return wrapper


@app.route('/setup', methods=['POST'])
@require_json
def setup(data: Any) -> Any:
    try:
        model_id = data['model_id']
        env_name = data['env_class']

        # 分配pid
        pid = str(uuid.uuid4())
        clients[pid] = time.time()
        AIes[pid] = AIServer(env_name, model_id, 1000)
        return jsonify({"status": "success", 'pid': pid})
    except Exception as e:
        print(f'Failed to setup AI: {e}')
        return jsonify({"error": f'Failed to setup AI:{e}'}), 400


@app.route('/make_move', methods=['POST'])
@require_json
def make_move(data: Any) -> Any:
    try:
        pid = data['pid']
        if pid not in AIes:
            return jsonify({"error": "Player has not been setup properly!"}), 400

        clients[pid] = time.time()
        state = np.array(data['array'], dtype=np.float32)
        last_action = data['action']
        player_to_move = data['player_to_move']
        action = AIes[pid].get_action(state, last_action, player_to_move)
        return jsonify({"status": "success", "action": action})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f'Failed to make move:{e}'}), 400


@app.route('/reset', methods=['POST'])
@require_json
def reset(data: Any) -> Any:
    try:
        pid = data['pid']
        if pid not in AIes:
            return jsonify({"error": "Player has not been setup properly!"}), 400

        clients[pid] = time.time()
        AIes[pid].reset()
        return jsonify({"status": 'success'})
    except Exception as e:
        return jsonify({"error": f'Failed to reset:{e}'}), 400


@app.route('/heartbeat', methods=['POST'])
@require_json
def heartbeat(data: Any) -> Any:
    try:
        pid = data['pid']
        if pid not in AIes:
            return jsonify({"error": "Player has not been setup properly!"}), 400

        clients[pid] = time.time()
        return jsonify({"status": 'success'})
    except Exception as e:
        return jsonify({"error": f'Failed to reset:{e}'}), 400


def cleanup_dead_clients(timeout=60):
    """定期检查清理掉失去连接的客户端"""
    while True:
        now = time.time()
        to_delete = []
        for pid, last_beat in clients.items():
            if now - last_beat > timeout:
                to_delete.append(pid)

        for pid in to_delete:
            clients.pop(pid)
            player = AIes.pop(pid)
            player.shutdown()  # 清理player中的mcts

        time.sleep(timeout)


threading.Thread(target=cleanup_dead_clients, daemon=True).start()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
