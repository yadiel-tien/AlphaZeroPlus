from typing import TypedDict, Literal

from utils.types import EnvName


class ColorTheme(TypedDict):
    blue: list[str]
    green: list[str]
    red: list[str]
    orange: list[str]
    grey: list[str]
    black: list[str]


cwd = '/home/bigger/projects/five_in_a_row'
color_key = Literal['blue', 'green', 'red', 'orange', 'grey', 'black']


class GameConfig(TypedDict):
    screen_size: tuple[int, int]
    grid_size: float
    n_filter: int
    n_cells: int
    n_res_blocks: int
    n_channels: int
    n_actions: int
    img_path: str
    state_shape: tuple[int, ...]
    tao_switch_steps: int


class AppConfig(TypedDict):
    color_themes: ColorTheme
    ChineseChess: GameConfig
    Gomoku: GameConfig
    dirichlet: float
    base_url: str
    device: str
    game_name: EnvName
    socket_path_prefix: str
    hub_socket_path: str
    train_socket_path: str
    data_dir: str
    log_dir: str
    buffer_name: str
    best_index_name: str
    ema_name: str
    rates_dir: str


CONFIG: AppConfig = {
    'color_themes': {
        'blue': ['#007bff', '#0056b3', '#004085', '#0056b3'],
        'green': ['#28a745', '#218838', '#1e7e34', '#218838'],
        'red': ['#dc3545', '#c82333', '#bd2130', '#c82333'],
        'orange': ['#fd7e14', '#e67e00', '#d45d02', '#e67e00'],
        'grey': ['#6c757d', '#5a6268', '#343a40', '#5a6268'],
        'black': ['#000000', '#333333', '#555555', '#333333']
    },
    'ChineseChess': {
        'screen_size': (600, 800),
        'grid_size': 54,
        'n_filter': 256,
        'state_shape': (10, 9, 20),
        'n_cells': 10 * 9,
        'n_res_blocks': 15,
        'n_channels': 20,
        'n_actions': 2086,
        'img_path': './graphics/chess/board.jpeg',
        'tao_switch_steps': 20
    },
    'Gomoku': {
        'screen_size': (600, 800),
        'grid_size': 35.2857,
        'n_filter': 256,
        'state_shape': (15, 15, 2),
        'n_cells': 15 * 15,
        'n_res_blocks': 10,
        'n_channels': 2,
        'n_actions': 15 * 15,
        'img_path': './graphics/gomoku/board.jpeg',
        'tao_switch_steps': 10
    },
    'data_dir': './data/',
    'dirichlet': 0.2,
    'base_url': 'http://192.168.0.126:5000/',
    'device': 'cuda:0',
    'game_name': 'Gomoku',
    'socket_path_prefix': './inference/socks/',
    'hub_socket_path': './inference/socks/hub.sock',
    'train_socket_path': cwd + '/inference/socks/train.sock',
    'log_dir': './logs/',
    'buffer_name': 'buffer.pkl',
    'best_index_name': 'best_index.pkl',
    'ema_name': 'ema.pkl',
    'rates_dir': './rates/',
}
game_name = CONFIG['game_name']
settings = CONFIG[game_name]
