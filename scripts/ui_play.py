import sys
import pygame
from core.utils.types import EnvName
from apps.ui.chess import ChineseChessUI
from apps.ui.gomoku import GomokuUI
from core.utils.config import CONFIG
from core.player.human import Human
from core.player.ai_client import AIClient
from apps.ui.launcher import LauncherUI

class Game:
    def __init__(self, game_name: EnvName, model1_idx: int, model2_idx: int) -> None:
        self.game_name = game_name
        self.settings = CONFIG[game_name]
        self.screen = pygame.display.set_mode(self.settings['screen_size'])
        pygame.display.set_caption(self.game_name)
        
        if model1_idx == -1 and model2_idx == -1:
            self.players = [Human(self.game_name), Human(self.game_name)]
        elif model2_idx == -1:
            self.players = [Human(self.game_name), AIClient(model1_idx, self.game_name)]
        else:
            self.players = [AIClient(model1_idx, self.game_name), AIClient(model2_idx, self.game_name)]
            
        if self.game_name == 'Gomoku':
            self.board = GomokuUI(self.players)
        else:
            self.board = ChineseChessUI(players=self.players)

    def play(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'QUIT'
                self.board.handle_input(event)

            self.board.update()
            self.board.draw()
            pygame.display.update()
            
            if self.board.status == 'returning':
                return 'LAUNCHER'
        return 'QUIT'

    def shutdown(self):
        for player in self.board.players:
            if hasattr(player, 'shutdown'):
                player.shutdown()

def run_launcher():
    # Launcher setup should be inside the main loop if we want to restart it
    screen = pygame.display.set_mode((600, 800))
    pygame.display.set_caption("ZenZero Launcher")
    launcher = LauncherUI()
    
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None
            launcher.handle_input(event)
        
        if launcher.state == 'FINISHED':
            return launcher.selected_game, launcher.selected_model1, launcher.selected_model2
            
        launcher.draw()
        pygame.display.update()
        clock.tick(60)
    return None, None, None

if __name__ == '__main__':
    pygame.init()
    
    while True:
        # Start with launcher
        selected_game, m1, m2 = run_launcher()
        
        if selected_game is None:
            break
        
        # Initialize the actual game
        game = Game(selected_game, m1, m2)
        result = game.play()
        game.shutdown()
        
        if result == 'QUIT':
            break
            
    pygame.quit()
    sys.exit()
