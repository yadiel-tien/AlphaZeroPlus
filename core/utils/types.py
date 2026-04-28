from enum import IntEnum
from typing import TypeAlias, Callable, Literal

from numpy.typing import NDArray

ChessMove: TypeAlias = tuple[int, int, int, int]
PieceMoveFunc: TypeAlias = Callable[[NDArray, int, int], list[tuple[int, int]]]
GomokuMove: TypeAlias = tuple[int, int]


class GameResult(IntEnum):
    ONGOING = 2
    WIN = 1
    DRAW = 0
    LOSE = -1


EnvName: TypeAlias = Literal['Gomoku', 'ChineseChess']
