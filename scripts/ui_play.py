import sys

sys.path.append('/Users/flashlight/PycharmProjects/FiveInARow/')
from utils.types import EnvName
import pygame

from ui.chess import ChineseChessUI
from ui.gomoku import GomokuUI
from utils.config import settings
from player.human import Human
from player.ai_client import AIClient

game_name: EnvName = 'Gomoku'


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(settings['screen_size'])
        pygame.display.set_caption(game_name)
        players = [Human(game_name), AIClient(311, game_name)]
        if game_name == 'Gomoku':
            self.board = GomokuUI(players)
        else:
            self.board = ChineseChessUI(players=players)

    def play(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.board.handle_input(event)

            self.board.update()
            self.board.draw()
            pygame.display.update()


if __name__ == '__main__':
    game = Game()
    game.play()
    pygame.quit()
    sys.exit()
