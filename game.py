"""
game.py - Game logic for 5x5 TicTacToe (4-in-a-row to win)
SWE3052-41 Homework 1: Adversarial Search
"""

import copy


class Board:
    """Represents a 5x5 TicTacToe board."""

    SIZE = 5
    WIN_LENGTH = 4

    def __init__(self):
        self.grid = [[None] * self.SIZE for _ in range(self.SIZE)]
        self.current_player = 'X'
        self.move_count = 0

    def copy(self):
        new_board = Board()
        new_board.grid = copy.deepcopy(self.grid)
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        return new_board

    def get_legal_moves(self):
        """Return list of (row, col) tuples for empty cells."""
        return [
            (r, c)
            for r in range(self.SIZE)
            for c in range(self.SIZE)
            if self.grid[r][c] is None
        ]

    def make_move(self, row, col, player=None):
        """Place a mark at (row, col). Uses current_player if player not specified."""
        if player is None:
            player = self.current_player
        if self.grid[row][col] is not None:
            raise ValueError(f"Cell ({row}, {col}) is already occupied.")
        self.grid[row][col] = player
        self.move_count += 1
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def undo_move(self, row, col):
        """Remove the mark at (row, col) and revert player turn."""
        self.grid[row][col] = None
        self.move_count -= 1
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        """
        Check if there is a winner.
        Returns 'X', 'O', or None.
        """
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.grid[r][c] is None:
                    continue
                player = self.grid[r][c]
                # Check all 4 directions from this cell
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if self._check_line(r, c, dr, dc, player):
                        return player
        return None

    def _check_line(self, r, c, dr, dc, player):
        """Check if WIN_LENGTH consecutive marks exist starting at (r,c) in direction (dr,dc)."""
        for i in range(self.WIN_LENGTH):
            nr, nc = r + dr * i, c + dc * i
            if not (0 <= nr < self.SIZE and 0 <= nc < self.SIZE):
                return False
            if self.grid[nr][nc] != player:
                return False
        return True

    def is_draw(self):
        """Return True if board is full and no winner."""
        return self.move_count == self.SIZE * self.SIZE and self.check_winner() is None

    def is_terminal(self):
        """Return True if game is over (win or draw)."""
        return self.check_winner() is not None or self.is_draw()

    def display(self):
        """Print the current board to console."""
        col_labels = "  " + " ".join(str(c) for c in range(self.SIZE))
        print(col_labels)
        print("  " + "-" * (self.SIZE * 2 - 1))
        for r in range(self.SIZE):
            row_str = f"{r}|"
            for c in range(self.SIZE):
                cell = self.grid[r][c]
                row_str += (cell if cell is not None else ".") + "|"
            print(row_str)
        print()


class Game:
    """Manages a game between two agents."""

    def __init__(self, agent_x, agent_o, verbose=True):
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.verbose = verbose

    def play(self):
        """
        Run a full game. Returns a dict with:
          - winner: 'X', 'O', or None (draw)
          - moves: list of (player, row, col)
          - node_expansions: from AlphaBetaAgent if applicable
          - computation_time: from AlphaBetaAgent if applicable
          - game_length: total number of moves
        """
        board = Board()
        moves = []

        # Map player symbol to agent
        agents = {'X': self.agent_x, 'O': self.agent_o}

        if self.verbose:
            print("=" * 40)
            print(f"  New Game: X={type(self.agent_x).__name__} vs O={type(self.agent_o).__name__}")
            print("=" * 40)
            board.display()

        while not board.is_terminal():
            player = board.current_player
            agent = agents[player]
            move = agent.choose_move(board)

            if move is None:
                # Fallback: random move (should not normally happen)
                import random
                move = random.choice(board.get_legal_moves())

            row, col = move
            board.make_move(row, col, player)
            moves.append((player, row, col))

            if self.verbose:
                print(f"  Player {player} plays ({row}, {col})")
                board.display()

        winner = board.check_winner()

        # Collect stats from AlphaBetaAgent if present
        node_expansions = 0
        computation_time = 0.0
        for ag in [self.agent_x, self.agent_o]:
            if hasattr(ag, 'total_nodes'):
                node_expansions += ag.total_nodes
            if hasattr(ag, 'total_time'):
                computation_time += ag.total_time

        if self.verbose:
            if winner:
                print(f"  *** Player {winner} WINS! ***")
            else:
                print("  *** DRAW! ***")
            #print(f"  Game length: {len(moves)} moves")
            if node_expansions:
                print(f"  Node expansions: {node_expansions}")
            if computation_time:
                print(f"  Computation time: {computation_time:.3f}s")
            print()

        return {
            'winner': winner,
            'moves': moves,
            'node_expansions': node_expansions,
            'computation_time': computation_time,
            'game_length': len(moves),
        }
