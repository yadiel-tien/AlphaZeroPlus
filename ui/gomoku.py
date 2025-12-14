import numpy as np
import pygame
from .baseui import GameUI
from utils.config import GameConfig, CONFIG

from env.gomoku import Gomoku
from typing import cast
from player.player import Player
from player.human import Human

settings: GameConfig = CONFIG['Gomoku']


class GomokuUI(GameUI):
    def __init__(self, players: list['Player'], rows=15, columns=15):
        super().__init__(Gomoku(rows, columns), players, settings['img_path'])
        self.black_piece = pygame.image.load('./graphics/gomoku/black.png')
        self.white_piece = pygame.image.load('./graphics/gomoku/white.png')
        self.mark_pic = pygame.image.load('./graphics/gomoku/circle.png')
        self.cursor_pic = pygame.image.load('./graphics/gomoku/cursor.png')
        self.env = cast(Gomoku, self.env)
        self.settings = settings

    def handle_human_input(self) -> None:
        player = cast(Human, self.players[self.env.player_to_move])
        if player.selected_grid is None:
            return
        action = self.env.move2action(player.selected_grid)
        if action in self.env.valid_actions:
            player.pending_action = action

    def play_place_sound(self, action: int) -> None:
        self.piece_sound.play()

    def draw(self):
        self.screen.fill('#DDDDBB')
        self.screen.blit(self.image, self.rect)
        self.draw_boundary()
        self.draw_pieces()
        self.draw_last_mark()

        if self.status == 'finished':
            self.draw_step_mark()
            self.draw_victory_badge()
            if self.timers[self.env.player_to_move].remain > 0:
                self.draw_victory_stones()
            self.start_btn.draw()
            self.reverse_player_btn.draw()
        elif self.status == 'new':
            self.draw_new_game_title()
            self.start_btn.draw()
            self.reverse_player_btn.draw()
        else:
            self.draw_player()
            self.draw_cursor()
            self.resign_btn.draw()

    def draw_pieces(self):

        # 获取所有白棋和黑棋的位置
        white_positions = np.argwhere(self.env.state[1 - self.env.player_to_move])
        black_positions = np.argwhere(self.env.state[self.env.player_to_move])

        # 绘制所有白棋
        for pos in white_positions:
            x, y = self._grid2pos(pos)
            self.screen.blit(self.white_piece, (x, y))

        # 绘制所有黑棋
        for pos in black_positions:
            x, y = self._grid2pos(pos)
            self.screen.blit(self.black_piece, (x, y))

    def draw_step_mark(self):
        font = pygame.font.Font(None, 16)
        for idx, (action, mark) in enumerate(self.history):
            color = 'black' if mark else 'white'  # 黑棋则白字，白棋则黑字
            text = font.render(str(idx + 1), True, color)
            x, y = self._grid2pos(self.env.action2move(action))
            rect = text.get_rect(center=(x + 17, y + 17))
            self.screen.blit(text, rect.topleft)

    def draw_last_mark(self):
        if self.history:
            action, _ = self.history[-1]
            move = self.env.action2move(action)
            x, y = self._grid2pos(move)
            self.screen.blit(self.mark_pic, (x, y))

    def draw_victory_badge(self):
        winner = 'black' if self.env.winner == 0 else 'white' if self.env.winner == 1 else 'draw'
        path = f'graphics/gomoku/{winner}_win.png'
        self.draw_victory(path)

    def draw_victory_stones(self):
        for grid in self.env.get_win_stones(self.env.state, self.env.last_action):
            x, y = self._grid2pos(grid)
            self.screen.blit(self.mark_pic, (x, y))

    def draw_cursor(self):
        if self.cursor_grid is not None:
            x, y = self._grid2pos(self.cursor_grid)
            self.screen.blit(self.cursor_pic, (x, y))

    def draw_boundary(self):
        if self.env.shape[1:] == (15, 15):
            return
        # 外部轮廓
        x0 = -7 * settings['grid_size'] + self.rect.centerx - 17
        y0 = -7 * settings['grid_size'] + self.rect.centery - 17
        w0 = settings['grid_size'] * 15
        h0 = settings['grid_size'] * 15
        # 可下棋区域
        x, y = self._grid2pos((0, 0))
        _, h, w = self.env.shape
        w *= settings['grid_size']
        h *= settings['grid_size']
        overlay = pygame.Surface((w0, h0), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # 半透明黑色 (RGBA)
        pygame.draw.rect(overlay, 'green', (x - x0 - 1, y - y0 - 1, w + 2, h + 2), 2)
        # 在半透明表面上绘制一个完全透明的矩形
        pygame.draw.rect(overlay, (0, 0, 0, 0), (x - x0, y - y0, w, h))
        self.screen.blit(overlay, (x0, y0))
