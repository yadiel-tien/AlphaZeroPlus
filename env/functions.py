from utils.types import EnvName
from .env import BaseEnv
from .chess import ChineseChess
from .gomoku import Gomoku


def get_class(name: EnvName) -> type[BaseEnv]:
    """根据env类名返回类本身"""
    if name == 'Gomoku':
        return Gomoku
    elif name == 'ChineseChess':
        ChineseChess.init_class_dicts()
        return ChineseChess
    else:
        raise ValueError('Unknown game class')
