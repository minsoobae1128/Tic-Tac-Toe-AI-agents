"""
agents.py - Agent implementations for 5x5 TicTacToe (4-in-a-row)
SWE3052-41 Homework 1: Adversarial Search

Agents:
  - AlphaBetaAgent : Minimax + Alpha-Beta pruning with a custom evaluation function
  - RandomAgent    : Selects a random legal move
  - NoisyHeuristicAgent : Rule-based heuristic with epsilon-random noise
"""

import random
import time
import math


# ---------------------------------------------------------------------------
# AlphaBetaAgent
# ---------------------------------------------------------------------------

class AlphaBetaAgent:
    """
    Minimax agent with alpha-beta pruning, depth-limited search,
    move ordering, and a 3-second per-move time limit.

    Evaluation Function Design:
    ----------------------------
    The evaluation function scores a board position from the perspective
    of `self.player`. It combines four components:

    1. **Threat scoring** (most important):
       For every possible window of WIN_LENGTH=4 consecutive cells
       (horizontal, vertical, diagonal), count:
         - 4-in-a-row  → ±1000000 (terminal win/loss, handled by minimax)
         - 3-in-a-row + 1 empty → ±500  (immediate threat)
         - 2-in-a-row + 2 empty → ±50   (growing threat)
         - 1-in-a-row + 3 empty → ±5    (potential)
       Windows containing marks of BOTH players are scored 0 (blocked).

    2. **Center control**:
       Cells closer to the board center are more strategically valuable
       (more lines pass through them). We add a small bonus proportional
       to (2 - distance_from_center) clamped to [0, 2].

    3. **Blocking priority**:
       Opponent 3-in-a-row threats are weighted 1.5× their attack value
       to ensure the agent prioritises defence when needed.

    4. **Move ordering** (speeds up pruning):
       Moves are sorted by a shallow evaluation before recursion so that
       better moves are explored first, maximising cut-offs.
    """

    WIN_LENGTH = 4
    SIZE = 5
    TIME_LIMIT = 3.0   # seconds per move
    DEPTH_LIMIT = 4    # chosen depth (see report for analysis)

    # Scores for n-in-a-row within an open window
    SCORE_TABLE = {4: 1_000_000, 3: 500, 2: 50, 1: 5}

    # Center cell for proximity bonus
    CENTER = (2, 2)
    CENTER_BONUS_WEIGHT = 3

    def __init__(self, player='X', depth=None):
        self.player = player          # 'X' or 'O'
        self.opponent = 'O' if player == 'X' else 'X'
        self.depth = depth if depth is not None else self.DEPTH_LIMIT

        # Stats reset each game
        self.total_nodes = 0
        self.total_time = 0.0
        self._move_start = 0.0
        self._timeout = False

    def reset_stats(self):
        self.total_nodes = 0
        self.total_time = 0.0

    def choose_move(self, board):
        """Select the best move using alpha-beta pruning."""
        self._move_start = time.time()
        self._timeout = False

        legal = board.get_legal_moves()
        if not legal:
            return None

        # Order moves for better pruning
        legal = self._order_moves(board, legal, maximising=True)

        best_move = legal[0]
        best_val = -math.inf

        alpha = -math.inf
        beta = math.inf

        for move in legal:
            if time.time() - self._move_start >= self.TIME_LIMIT:
                self._timeout = True
                break

            row, col = move
            board.make_move(row, col, self.player)
            val = self._minimax(board, self.depth - 1, alpha, beta, maximising=False)
            board.undo_move(row, col)

            if val > best_val:
                best_val = val
                best_move = move

            alpha = max(alpha, best_val)

        elapsed = time.time() - self._move_start
        self.total_time += elapsed

        if self._timeout:
            # Fall back to the random
            return random.choice(board.get_legal_moves())

        return best_move

    def _minimax(self, board, depth, alpha, beta, maximising):
        """
        Recursive minimax with alpha-beta pruning.
        Returns the heuristic value of `board` from self.player's perspective.
        """
        self.total_nodes += 1

        winner = board.check_winner()
        if winner == self.player:
            return 1_000_000 + depth   # win sooner is better
        if winner == self.opponent:
            return -(1_000_000 + depth)
        if board.is_draw():
            return 0
        if depth == 0:
            return self._evaluate(board)
        if time.time() - self._move_start >= self.TIME_LIMIT:
            self._timeout = True
            return self._evaluate(board)

        legal = board.get_legal_moves()
        legal = self._order_moves(board, legal, maximising=maximising)

        if maximising:
            value = -math.inf
            for move in legal:
                board.make_move(move[0], move[1], self.player)
                value = max(value, self._minimax(board, depth - 1, alpha, beta, False))
                board.undo_move(move[0], move[1])
                alpha = max(alpha, value)
                if value >= beta:
                    break   # beta cut-off
            return value
        else:
            value = math.inf
            for move in legal:
                board.make_move(move[0], move[1], self.opponent)
                value = min(value, self._minimax(board, depth - 1, alpha, beta, True))
                board.undo_move(move[0], move[1])
                beta = min(beta, value)
                if value <= alpha:
                    break   # alpha cut-off
            return value

    # ------------------------------------------------------------------
    # Evaluation function
    # ------------------------------------------------------------------

    def _evaluate(self, board):
        """
        Heuristic board evaluation from self.player's perspective.
        Higher values are better for self.player.
        """
        score = 0

        # 1. Score all windows of length WIN_LENGTH
        for window, r, c, dr, dc in self._all_windows(board):
            score += self._score_window(window)

        # 2. Center control bonus
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                cell = board.grid[r][c]
                if cell is None:
                    continue
                dist = abs(r - self.CENTER[0]) + abs(c - self.CENTER[1])
                bonus = max(0, self.CENTER_BONUS_WEIGHT - dist)
                if cell == self.player:
                    score += bonus
                else:
                    score -= bonus

        return score

    def _score_window(self, window):
        """
        Score a window of WIN_LENGTH cells.
        Returns a positive value if good for self.player, negative if bad.
        """
        my_count = window.count(self.player)
        opp_count = window.count(self.opponent)
        empty_count = window.count(None)

        # Mixed window: neither player can complete this line
        if my_count > 0 and opp_count > 0:
            return 0

        if my_count > 0:
            base = self.SCORE_TABLE.get(my_count, 0)
            return base

        if opp_count > 0:
            base = self.SCORE_TABLE.get(opp_count, 0)
            # Weight opponent 3-threats more heavily (defensive priority)
            if opp_count == 3:
                base = int(base * 1.5)
            return -base

        return 0

    def _all_windows(self, board):
        """
        Generator yielding (window, r, c, dr, dc) for every
        possible WIN_LENGTH consecutive window on the board.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                for dr, dc in directions:
                    cells = []
                    valid = True
                    for i in range(self.WIN_LENGTH):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                            cells.append(board.grid[nr][nc])
                        else:
                            valid = False
                            break
                    if valid:
                        yield cells, r, c, dr, dc

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def _order_moves(self, board, moves, maximising):
        """
        Sort moves by a fast heuristic so better moves are tried first.
        This increases alpha-beta pruning efficiency.
        """
        def move_score(move):
            r, c = move
            player = self.player if maximising else self.opponent
            board.make_move(r, c, player)
            val = self._evaluate(board)
            board.undo_move(r, c)
            return val

        return sorted(moves, key=move_score, reverse=maximising)


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class RandomAgent:
    """Selects a uniformly random legal move."""

    def __init__(self, player='O'):
        self.player = player

    def choose_move(self, board):
        legal = board.get_legal_moves()
        if not legal:
            return None
        return random.choice(legal)


# ---------------------------------------------------------------------------
# NoisyHeuristicAgent
# ---------------------------------------------------------------------------

class NoisyHeuristicAgent:
    """
    Rule-based heuristic agent with epsilon-random noise (ε = 0.2).

    Heuristic rules (in priority order):
      1. **Immediate win**: If a move completes 4-in-a-row, take it.
      2. **Block opponent win**: If the opponent can win in one move, block it.
      3. **Positional preference**: Prefer center cells (lowest Manhattan
         distance to (2,2)), then adjacent cells, then edges.
      4. **Random fallback**: If all remaining options are equal, pick randomly.

    With probability ε = 0.2, the agent ignores all rules and plays randomly.
    """

    WIN_LENGTH = 4
    SIZE = 5
    EPSILON = 0.2

    def __init__(self, player='O'):
        self.player = player
        self.opponent = 'O' if player == 'X' else 'X'

    def choose_move(self, board):
        legal = board.get_legal_moves()
        if not legal:
            return None

        # Epsilon-random: ignore heuristic with probability EPSILON
        if random.random() < self.EPSILON:
            return random.choice(legal)

        # Rule 1: Win immediately if possible
        for move in legal:
            if self._would_win(board, move, self.player):
                return move

        # Rule 2: Block opponent's immediate win
        for move in legal:
            if self._would_win(board, move, self.opponent):
                return move

        # Rule 3: Positional preference (center proximity)
        center = (self.SIZE // 2, self.SIZE // 2)
        legal_sorted = sorted(
            legal,
            key=lambda m: abs(m[0] - center[0]) + abs(m[1] - center[1])
        )
        return legal_sorted[0]

    def _would_win(self, board, move, player):
        """Return True if placing `player` at `move` would win the game."""
        r, c = move
        board.grid[r][c] = player
        winner = board.check_winner()
        board.grid[r][c] = None
        return winner == player


# ---------------------------------------------------------------------------
# QLearningAgent
# ---------------------------------------------------------------------------
import pickle

class QLearningAgent:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.
    Uses a sparse Q-table to store state-action values.
    """
    def __init__(self, player='X', alpha=0.1, gamma=0.95, epsilon=1.0, min_epsilon=0.05, epsilon_decay=0.99995):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # Sparse Q-table: maps state (tuple of tuples) to a dict of {action: q_value}
        self.q_table = {}

    def _get_q_values(self, state, legal_actions):
        """Returns the dictionary of Q-values for a state, initializing missing actions to 0.0."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in legal_actions}
        else:
            for a in legal_actions:
                if a not in self.q_table[state]:
                    self.q_table[state][a] = 0.0
        return self.q_table[state]

    def select_action(self, state, legal_actions, epsilon=None):
        """Selects an action using epsilon-greedy policy."""
        if not legal_actions:
            return None
            
        current_epsilon = epsilon if epsilon is not None else self.epsilon
        q_vals = self._get_q_values(state, legal_actions)
        
        if random.random() < current_epsilon:
            return random.choice(legal_actions)
        else:
            # Greedy choice: pick action with max Q-value, break ties randomly
            max_q = max(q_vals.values())
            best_actions = [a for a, q in q_vals.items() if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_legal_actions, done):
        """Performs the tabular Q-learning update step."""
        q_vals = self._get_q_values(state, [action])
        current_q = q_vals[action]
        
        if done:
            target = reward
        else:
            next_q_vals = self._get_q_values(next_state, next_legal_actions)
            max_next_q = max(next_q_vals.values()) if next_legal_actions else 0.0
            target = reward + self.gamma * max_next_q
        
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def save_q_table(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)