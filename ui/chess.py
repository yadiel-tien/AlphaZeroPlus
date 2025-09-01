from typing import TypeAlias, Literal, cast

import pygame

from .baseui import GameUI
from utils.config import GameConfig, CONFIG
from player.human import Human
from env.chess import ChineseChess

Mark: TypeAlias = Literal['green_dot', 'red_dot', 'blue_circle', 'green_circle', 'low_shadow', 'high_shadow']
settings: GameConfig = CONFIG['ChineseChess']


class ChineseChessUI(GameUI):

    def __init__(self, players):
        super().__init__(ChineseChess(), players, settings['img_path'])
        self.piece_pics: dict[int, pygame.Surface] = {}
        self.mark_pics: dict[Mark, pygame.Surface] = {}
        self.init_resource()
        self.place_sound = pygame.mixer.Sound('./sound/piece_down.mp3')
        self.capture_sound = pygame.mixer.Sound('./sound/capture.mp3')
        self.check_sound = pygame.mixer.Sound('./sound/check.mp3')
        self.checkmate_sound = pygame.mixer.Sound('./sound/checkmate.mp3')
        self.image = pygame.transform.scale(self.image, (486, 540))
        self.rect = self.image.get_rect(center=self.screen.get_rect().center)
        self.env = cast(ChineseChess, self.env)
        self.selected_pos: tuple[int, int] | None = None
        self.settings = settings
        self.check_buffer = {'action': -1, 'checkmate': False, 'check': False, 'red_dot': {}}

    def init_resource(self) -> None:
        """加载棋子和标记图片"""
        for i in range(14):
            pic = pygame.image.load(f'graphics/chess/piece{i}.png')
            pic = pygame.transform.smoothscale(pic, (65, 65))
            self.piece_pics[i] = pic

        marks: tuple[Mark, ...] = ('red_dot', 'green_dot', 'blue_circle', 'green_circle', 'low_shadow', 'high_shadow')
        for mark in marks:
            pic = pygame.image.load(f'graphics/chess/{mark}.png')
            self.mark_pics[mark] = pygame.transform.smoothscale(pic, (65, 65))
        self.mark_pics['high_shadow'] = pygame.transform.smoothscale_by(self.mark_pics['high_shadow'], 1.6)

    def handle_human_input(self) -> None:
        player = cast(Human, self.players[self.env.player_to_move])
        if player.selected_grid is None:
            return

        if self.selected_pos:  # 已选择棋子
            # 根据已选择棋子和目标位置生成动作
            move = self.selected_pos + player.selected_grid
            action = self.env.move2action(move)
            if action in self.env.valid_actions:
                player.pending_action = action
                self.place_sound.play()
            self.selected_pos = None
            if player.pending_action == -1:
                self.piece_sound.play()
        else:
            # 选择棋子
            chosen_piece = self.env.state[player.selected_grid + (0,)]
            if chosen_piece == -1:
                return
            is_red_piece = (0 <= chosen_piece < 7)
            is_red_turn = self.env.player_to_move == 0
            if is_red_piece == is_red_turn:
                self.selected_pos = player.selected_grid
                self.piece_sound.play()

    def play_place_sound(self, action: int) -> None:
        """执行action时播放的音效"""
        if self.check_buffer['action'] != action:
            self.check_buffer['action'] = action
            self.check_buffer['checkmate'] = self.env.is_checkmate(self.env.state, self.env.player_to_move)
            self.check_buffer['check'] = self.env.is_check(self.env.state, self.env.player_to_move)

        # 绝杀，对方无论怎么走都输
        if self.check_buffer['checkmate']:
            self.checkmate_sound.play()
            return

        # 将军
        if self.check_buffer['check']:
            self.check_sound.play()
            return

        # 吃子
        _, _, to_r, to_c = self.env.action2move(action)
        target = self.env.state[to_r, to_c, 1]
        if target not in [-1, 4, 11]:
            self.capture_sound.play()

    def draw(self) -> None:
        self.screen.fill('#DDDDBB')
        self.screen.blit(self.image, self.rect)
        self.draw_last_mark()
        self.draw_dot_mark()
        self.draw_pieces()
        self.draw_select_piece()
        if self.status == 'finished':
            self.draw_victory_badge()
            self.start_btn.draw()
            self.reverse_player_btn.draw()
        elif self.status == 'new':
            self.draw_new_game_title()
            self.start_btn.draw()
            self.reverse_player_btn.draw()
        else:
            self.draw_player()

    def draw_victory_badge(self) -> None:
        winner = 'red' if self.env.winner == 0 else 'black' if self.env.winner == 1 else 'draw'
        path = f'graphics/chess/{winner}_win.png'
        self.draw_victory(path)

    def draw_pieces(self) -> None:
        """绘制棋子"""
        for row in range(10):
            for col in range(9):
                piece = int(self.env.state[row, col, 0])
                if piece != -1 and (row, col) != self.selected_pos:
                    x, y = self._grid2pos((row, col))
                    pic = self.piece_pics[piece]
                    self.screen.blit(self.mark_pics['low_shadow'], (x, y))
                    self.screen.blit(pic, (x, y))

    def draw_select_piece(self) -> None:
        """选中的棋子周围绘制绿色圆圈"""
        if self.selected_pos:
            row, col = self.selected_pos
            piece = int(self.env.state[row, col, 0])
            piece_pic = self.piece_pics[piece]
            x, y = self._grid2pos((row, col))
            rect = piece_pic.get_rect(topleft=(x, y))
            piece_pic = pygame.transform.smoothscale_by(piece_pic, 1.2)
            rect = piece_pic.get_rect(center=rect.center)
            shadow_rect = self.mark_pics['high_shadow'].get_rect(center=rect.center)
            self.screen.blit(self.mark_pics['high_shadow'], shadow_rect)
            self.screen.blit(piece_pic, rect)

    def draw_last_mark(self) -> None:
        """最后走的棋子周围绘制蓝色圆圈"""
        if self.history:
            action, _ = self.history[-1]
            _, _, to_r, to_c = self.env.action2move(action)
            grid = to_r, to_c
            x, y = self._grid2pos(grid)
            self.screen.blit(self.mark_pics['blue_circle'], (x, y))

    def draw_dot_mark(self) -> None:
        """用来指示所有可走棋步"""
        if self.selected_pos:
            is_red_dot = self.check_buffer['red_dot']
            piece = int(self.env.state[self.selected_pos + (0,)])
            grids = self.env.dest_func[piece](self.env.state, *self.selected_pos)
            for grid in grids:
                x, y = self._grid2pos(grid)
                # 模拟行棋，如果导致自身被将，则标红，否则标绿
                move = self.selected_pos + grid
                action = self.env.move2action(move)
                if len(is_red_dot) == len(grids):  # 已有缓存，避免重复计算
                    if is_red_dot[action]:
                        self.screen.blit(self.mark_pics['red_dot'], (x, y))
                    else:
                        self.screen.blit(self.mark_pics['green_dot'], (x, y))
                else:  # 重建缓存
                    new_state = self.env.virtual_step(self.env.state, action)
                    is_red_dot[action] = self.env.is_check(new_state, self.env.player_to_move)
        else:
            self.check_buffer['red_dot'] = {}

    def _grid2pos(self, grid: tuple[int, int]) -> tuple[int, int]:
        """调节位置偏差"""
        x, y = super()._grid2pos(grid)
        return x - 4 - x // 200, y - y // 100
