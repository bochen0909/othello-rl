"""
Human vs Agent Othello game (GUI).
"""

from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from aip_rl.othello.play_common import build_arg_parser, select_opponent

try:
    import pygame
except ImportError:
    print("Error: pygame is required for the GUI version.")
    print("Install with: pip install pygame")
    print("Or with poetry: poetry install --extras gui")
    sys.exit(1)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
DARK_GREEN = (0, 100, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
ORANGE = (255, 165, 0)

# Board settings
CELL_SIZE = 70
BOARD_SIZE = 8
MARGIN = 10
INFO_HEIGHT = 150
WINDOW_WIDTH = CELL_SIZE * BOARD_SIZE + 2 * MARGIN
WINDOW_HEIGHT = CELL_SIZE * BOARD_SIZE + 2 * MARGIN + INFO_HEIGHT


class OthelloGUI:
    """GUI for Othello game using Pygame."""

    def __init__(self, opponent_policy, human_color="black"):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Othello - Human vs Agent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.env = gym.make(
            "Othello-v0",
            opponent=opponent_policy,
            reward_mode="sparse",
            invalid_move_mode="error",
            render_mode=None,
            start_player=human_color,
        )
        self.opponent_policy = opponent_policy
        self.human_color = human_color
        self.human_is_black = human_color == "black"

        self.obs, self.info = self.env.reset()
        self.game_over = False
        self.winner = None
        self.waiting_for_opponent = False
        self.message = "Your turn!"
        self.last_move = None
        self.opponent_move_time = 0
        self.show_last_move_duration = 1500

        if not self.is_human_turn():
            board_before = self.get_board_state().copy()
            self.env.unwrapped._execute_opponent_move()
            self.info = self.env.unwrapped._get_info()
            self.last_move = self.find_last_move(
                board_before, self.get_board_state()
            )
            self.opponent_move_time = pygame.time.get_ticks()
            if self.env.unwrapped.game.get_winner() != 3:
                self.game_over = True
                self.winner = self.env.unwrapped.game.get_winner()
            self.message = "Your turn!"

    def get_board_state(self):
        """Extract board state from observation."""
        unwrapped = self.env.unwrapped
        return unwrapped.game.get_board()

    def get_valid_moves(self):
        """Get valid moves from action mask."""
        return self.info.get("action_mask", np.zeros(64, dtype=bool))

    def board_to_screen(self, row, col):
        """Convert board coordinates to screen coordinates."""
        x = MARGIN + col * CELL_SIZE
        y = MARGIN + row * CELL_SIZE
        return x, y

    def screen_to_board(self, x, y):
        """Convert screen coordinates to board coordinates."""
        if x < MARGIN or y < MARGIN:
            return None, None
        col = (x - MARGIN) // CELL_SIZE
        row = (y - MARGIN) // CELL_SIZE
        if row >= BOARD_SIZE or col >= BOARD_SIZE:
            return None, None
        return row, col

    def draw_board(self):
        """Draw the Othello board."""
        self.screen.fill(DARK_GREEN)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x, y = self.board_to_screen(row, col)
                pygame.draw.rect(
                    self.screen, GREEN, (x, y, CELL_SIZE, CELL_SIZE)
                )
                pygame.draw.rect(
                    self.screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1
                )

        if self.last_move is not None:
            elapsed = pygame.time.get_ticks() - self.opponent_move_time
            if elapsed < self.show_last_move_duration:
                row, col = self.last_move // 8, self.last_move % 8
                x, y = self.board_to_screen(row, col)
                pygame.draw.rect(
                    self.screen, ORANGE, (x, y, CELL_SIZE, CELL_SIZE), 5
                )

        board = self.get_board_state()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row, col]
                if piece != 0:
                    x, y = self.board_to_screen(row, col)
                    center_x = x + CELL_SIZE // 2
                    center_y = y + CELL_SIZE // 2
                    radius = CELL_SIZE // 2 - 5

                    color = BLACK if piece == 1 else WHITE
                    pygame.draw.circle(
                        self.screen, color, (center_x, center_y), radius
                    )
                    pygame.draw.circle(
                        self.screen, GRAY, (center_x, center_y), radius, 2
                    )

        if not self.game_over and self.is_human_turn():
            valid_moves = self.get_valid_moves()
            for idx in range(64):
                if valid_moves[idx]:
                    row, col = idx // 8, idx % 8
                    x, y = self.board_to_screen(row, col)
                    center_x = x + CELL_SIZE // 2
                    center_y = y + CELL_SIZE // 2
                    pygame.draw.circle(
                        self.screen, YELLOW, (center_x, center_y), 8
                    )

    def draw_info(self):
        """Draw game information panel."""
        info_y = CELL_SIZE * BOARD_SIZE + 2 * MARGIN

        pygame.draw.rect(
            self.screen, LIGHT_GRAY, (0, info_y, WINDOW_WIDTH, INFO_HEIGHT)
        )

        unwrapped = self.env.unwrapped
        black_count, white_count = unwrapped.game.get_piece_counts()

        score_text = f"● Black: {black_count}  ○ White: {white_count}"
        score_surface = self.font.render(score_text, True, BLACK)
        self.screen.blit(score_surface, (20, info_y + 10))

        player_text = f"You are: {self.human_color.upper()}"
        player_surface = self.small_font.render(player_text, True, BLACK)
        self.screen.blit(player_surface, (20, info_y + 50))

        opp_name = (
            self.opponent_policy
            if isinstance(self.opponent_policy, str)
            else "trained agent"
        )
        opp_text = f"Opponent: {opp_name}"
        opp_surface = self.small_font.render(opp_text, True, BLACK)
        self.screen.blit(opp_surface, (20, info_y + 75))

        if self.game_over:
            if self.winner == 2:
                msg = "DRAW!"
            elif (
                (self.winner == 0 and self.human_is_black)
                or (self.winner == 1 and not self.human_is_black)
            ):
                msg = "YOU WIN!"
                color = GREEN
            else:
                msg = "YOU LOSE!"
                color = RED
            msg_surface = self.font.render(msg, True, color)
        else:
            msg_surface = self.small_font.render(self.message, True, BLACK)

        self.screen.blit(msg_surface, (20, info_y + 105))

    def is_human_turn(self):
        """Check if it's the human player's turn."""
        current_player = self.info.get("current_player", 0)
        return (current_player == 0 and self.human_is_black) or (
            current_player == 1 and not self.human_is_black
        )

    def handle_click(self, pos):
        """Handle mouse click on the board."""
        if self.game_over or not self.is_human_turn():
            return

        x, y = pos
        row, col = self.screen_to_board(x, y)

        if row is None or col is None:
            return

        action = row * 8 + col
        valid_moves = self.get_valid_moves()

        if not valid_moves[action]:
            self.message = "Invalid move! Click a yellow dot."
            return

        try:
            board_before = self.get_board_state().copy()

            self.obs, reward, terminated, truncated, self.info = self.env.step(
                action
            )

            if terminated:
                self.game_over = True
                unwrapped = self.env.unwrapped
                self.winner = unwrapped.game.get_winner()
                self.last_move = None
            else:
                board_after = self.get_board_state()
                self.last_move = self.find_last_move(board_before, board_after)
                self.opponent_move_time = pygame.time.get_ticks()
                self.message = "Opponent's turn..."
                self.waiting_for_opponent = True

        except Exception as e:
            self.message = f"Error: {e}"

    def find_last_move(self, board_before, board_after):
        """Find the last move made by comparing board states."""
        opponent_color = 2 if self.human_is_black else 1
        new_pieces = np.where(
            (board_before == 0) & (board_after == opponent_color)
        )
        if len(new_pieces[0]) > 0:
            row, col = new_pieces[0][0], new_pieces[1][0]
            return row * 8 + col
        return None

    def update_opponent_move(self):
        """Process opponent's move (already handled by env)."""
        if self.waiting_for_opponent and not self.game_over:
            elapsed = pygame.time.get_ticks() - self.opponent_move_time
            if elapsed < 800:
                return

            self.waiting_for_opponent = False

            if self.is_human_turn():
                valid_moves = self.get_valid_moves()
                if np.any(valid_moves):
                    self.message = "Your turn!"
                else:
                    self.message = "No valid moves - turn passed"
            else:
                self.message = "Opponent's turn..."

    def run(self):
        """Main game loop."""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.obs, self.info = self.env.reset()
                        self.game_over = False
                        self.winner = None
                        self.waiting_for_opponent = False
                        self.message = "Your turn!"
                        self.last_move = None
                        self.opponent_move_time = 0

            if self.waiting_for_opponent:
                self.update_opponent_move()

            self.draw_board()
            self.draw_info()

            pygame.display.flip()
            self.clock.tick(30)

        self.env.close()
        pygame.quit()


def parse_args():
    """Parse command line arguments."""
    parser = build_arg_parser(
        description="Play Othello against an agent (GUI version)"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    opponent_policy = select_opponent(args)
    gui = OthelloGUI(opponent_policy, args.human_color)
    gui.run()


if __name__ == "__main__":
    main()
