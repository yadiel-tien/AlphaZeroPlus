from abc import ABC, abstractmethod
from typing import TypeAlias, Literal

import pygame

from env.chess import ChineseChess
from env.env import BaseEnv
from .button import Button
from utils.timer import Timer
from player.human import Human, Player

GameStatus: TypeAlias = Literal['new', 'playing', 'finished']


class GameUI(ABC):
    def __init__(self, env: BaseEnv, players: list[Player], img_path: str) -> None:

        self.env = env
        self.players = players
        self.status: GameStatus = 'new'

        self.piece_sound = pygame.mixer.Sound('./sound/place_stone.mp3')
        self.win_sound = pygame.mixer.Sound('./sound/win.mp3')
        pygame.mixer.music.load('./sound/bgm.mp3')
        pygame.mixer.music.set_volume(0.5)
        self.start_btn = Button("Start", self.start, (200, 680), color='green')
        self.reverse_player_btn = Button(f'First:{players[0].description}', self.reverse_player, pos=(200, 740),
                                         color='grey')
        self.resign_btn = Button('Resign', self.resign, (20, 20), (60, 30), color='red')
        self.timers = {0: Timer(limit=60000, func=self.time_up), 1: Timer(limit=60000, func=self.time_up)}
        self.history = []
        self.screen = pygame.display.get_surface()
        self.image = pygame.image.load(img_path)
        self.rect = self.image.get_rect(center=self.screen.get_rect().center)
        self.cursor_grid: tuple[int, int] | None = None
        self.settings = {}

    @property
    def is_view_flipped(self) -> bool:
        return False
        # return isinstance(self.players[1], Human) and not isinstance(self.players[0], Human)

    def handle_input(self, event: pygame.event.Event) -> None:
        if self.status == 'playing':
            player = self.players[self.env.player_to_move]

            if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
                self.cursor_grid = self._pos2grid(event.pos)

            if isinstance(player, Human):  # 处理人类玩家交互
                self.resign_btn.handle_input(event)  # 投降按钮
                if event.type == pygame.MOUSEBUTTONDOWN:
                    player.selected_grid = self.cursor_grid
                    self.handle_human_input()

        else:
            self.start_btn.handle_input(event)
            self.reverse_player_btn.handle_input(event)

    def update(self) -> None:
        if self.status == 'playing':
            player = self.players[self.env.player_to_move]
            action = player.pending_action
            if action != -1:
                self.history.append((action, self.env.player_to_move))
                self.env.step(action)
                # 临时添加，后期在step中进行
                if isinstance(self.env, ChineseChess):
                    if ChineseChess.is_checkmate(self.env.state, self.env.player_to_move):
                        self.env.winner = 1 - self.env.player_to_move
                        self.env.terminated = True
                self.play_place_sound(action)
                player.pending_action = -1
                if self.env.terminated or self.env.truncated:
                    self.set_win_status()
                else:
                    self.switch_side()
            else:
                player.update(self.env.state, self.env.last_action, self.env.player_to_move)

    @abstractmethod
    def handle_human_input(self) -> None:
        ...

    @abstractmethod
    def play_place_sound(self, action: int) -> None:
        """执行action时播放的音效"""

    def resign(self) -> None:
        """投降"""
        self.env.set_winner(1 - self.env.player_to_move)
        self.set_win_status()

    def switch_side(self):
        current_player = self.env.player_to_move
        prev_player = 1 - current_player
        # 重设timer
        self.timers[prev_player].reset()
        self.timers[current_player].activate()

    def time_up(self):
        for side, timer in self.timers.items():
            if timer.remain == 0:
                self.env.set_winner(1 - side)
                self.set_win_status()

    def set_win_status(self):
        self.status = 'finished'
        self.start_btn.text = "Restart"
        pygame.mixer.music.stop()
        self.win_sound.play()

    def start(self):
        if self.status == 'finished':
            self.env.reset()
            for timer in self.timers.values():
                timer.reset()
        # if isinstance(self.env, ChineseChess):
        #     self.env.random_opening()
        self.history = []
        self.status = 'playing'
        # 重设玩家
        for player in self.players:
            player.reset()
        # 当前玩家开始计时
        self.timers[self.env.player_to_move].activate()
        pygame.mixer.music.play()

    def reverse_player(self):
        self.players.reverse()
        self.reverse_player_btn.text = f'First: {self.players[0].description}'

    def draw_victory(self, path):
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (200, 200))
        x = self.rect.centerx
        y = 60
        rect = image.get_rect(center=(x, y))
        self.screen.blit(image, rect.topleft)
        font = pygame.font.Font(None, 36)
        text = font.render(f'winner:{self.players[1 - self.env.player_to_move].description}', True, (255, 255, 30))
        rect = text.get_rect(center=(x, y))
        self.screen.blit(text, rect.topleft)

    def draw_new_game_title(self):
        font = pygame.font.Font(None, 70)
        text = font.render('New Game', True, 'orange')
        x = self.rect.centerx
        y = 60
        rect = text.get_rect(center=(x, y))
        self.screen.blit(text, rect.topleft)

    def draw_player(self):
        self.timers[self.env.player_to_move].update()
        desc_vertical_positions = [60, 740] if self.is_view_flipped else [740, 60]
        sides = ["Red", "Black"]
        for i, player in enumerate(self.players):
            time_remain = self.timers[i].remain // 1000
            font = pygame.font.Font(None, 60)
            desc = font.render(f'{player.description}  {sides[i]}:{time_remain:02}', True, 'orange')
            desc_rect = desc.get_rect(center=(300, desc_vertical_positions[i]))
            self.screen.blit(desc, desc_rect.topleft)

            if hasattr(player, 'win_rate'):
                win_rate = player.win_rate
                font = pygame.font.Font(None, 30)
                win_rate_label = font.render(f'win rate:{win_rate:.2%} ', True, 'gray')
                self.screen.blit(win_rate_label, (220, desc_rect.bottom + 10))

    def _pos2grid(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """根据屏幕坐标，返回棋盘位置，超出棋盘返回None
        :param pos:屏幕坐标(x,y)
        :return 棋盘坐标(row,col)
        """
        center_x, center_y = self.screen.get_rect().center
        rows, columns, _ = self.env.shape
        grid_size = self.settings['grid_size']

        col: int = round((pos[0] - center_x) / grid_size + (columns - 1) / 2)
        row: int = round((pos[1] - center_y) / grid_size + (rows - 1) / 2)

        return (row, col) if 0 <= col < columns and 0 <= row < rows else None

    def _grid2pos(self, grid: tuple[int, int]) -> tuple[int, int]:
        """
        根据棋盘交叉点位置返回该位置top left坐标。即以交叉点为中心，grid_size为边长的矩形左上角端点坐标。
        :param grid:棋盘坐标(row,col)
        :return 交叉点top left屏幕坐标(x,y)
        """
        row, col = grid
        center_x, center_y = self.screen.get_rect().center
        rows, columns, _ = self.env.shape
        grid_size = self.settings['grid_size']

        x = (col - (columns - 1) / 2) * grid_size + center_x - grid_size // 2
        y = (row - (rows - 1) / 2) * grid_size + center_y - grid_size // 2
        return int(x), int(y)
