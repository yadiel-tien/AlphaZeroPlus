import signal
import sys
from typing import Callable


def register_sigint(cleanup_func: Callable[[], None]) -> None:
    """注册响应Ctrl+C，清理资源后退出，而不是直接中止"""

    def handle_sigint(sig, frame):
        print("\n[INFO] Caught Ctrl+C, shutting down...")
        cleanup_func()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
