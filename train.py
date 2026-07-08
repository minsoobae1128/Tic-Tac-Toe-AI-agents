import random
from game import Board
from agents import QLearningAgent, RandomAgent, NoisyHeuristicAgent

class TicTacToeRLEnv:
    """RL Wrapper around the HW1 Board to allow step() and reset() flows."""
    def __init__(self, q_player='X', opponent_agent=None, config="baseline"):
        self.q_player = q_player
        self.opp_player = 'O' if q_player == 'X' else 'X'
        self.opponent_agent = opponent_agent
        self.config = config
        
        # Ensure opponent agent uses the correct symbol
        if self.opponent_agent:
            self.opponent_agent.player = self.opp_player
            if hasattr(self.opponent_agent, 'opponent'):
                self.opponent_agent.opponent = self.q_player

    def get_state(self):
        """Returns a hashable tuple-of-tuples representation of the grid."""
        return tuple(tuple(row) for row in self.board.grid)

    def reset(self, q_starts=True):
        self.board = Board()
        # If opponent starts, they make the first move before returning state to Q-agent
        if not q_starts and self.opponent_agent:
            action = self.opponent_agent.choose_move(self.board)
            if action:
                self.board.make_move(action[0], action[1], self.opp_player)
        return self.get_state()

    def legal_actions(self):
        return self.board.get_legal_moves()

    def step(self, action):
        """Executes Q-agent move, checks win, then executes opponent move."""
        # 1. Q-Agent Turn
        self.board.make_move(action[0], action[1], self.q_player)
        winner = self.board.check_winner()
        
        if winner == self.q_player:
            return self.get_state(), 1.0, True, {'winner': winner}
        elif winner is None and self.board.is_draw():
            return self.get_state(), 0.0, True, {'winner': None}
            
        # 2. Opponent Turn
        opp_action = self.opponent_agent.choose_move(self.board)
        if opp_action:
            self.board.make_move(opp_action[0], opp_action[1], self.opp_player)
            
        winner = self.board.check_winner()
        if winner == self.opp_player:
            return self.get_state(), -1.0, True, {'winner': winner}
        elif winner is None and self.board.is_draw():
            return self.get_state(), 0.0, True, {'winner': None}
            
        # 3. Intermediate rewards for Config 2
        reward = 0.0
        if self.config == "improved":
            # Improvement: Small step penalty to encourage faster wins / longer survival
            reward = -0.01 
            
        return self.get_state(), reward, False, {'winner': None}


def train_agent(episodes, config, opponent, q_player='X'):
    """Trains a QLearningAgent against a specific opponent."""
    agent = QLearningAgent(player=q_player, alpha=0.1, gamma=0.95, epsilon=1.0)
    env = TicTacToeRLEnv(q_player=q_player, opponent_agent=opponent, config=config)
    
    print(f"Starting Training: {config.upper()} Configuration against {type(opponent).__name__}...")
    for ep in range(1, episodes + 1):
        # 50/50 starting split per assignment requirements
        q_starts = (ep % 2 == 0)
        state = env.reset(q_starts=q_starts)
        done = False
        
        while not done:
            legal = env.legal_actions()
            action = agent.select_action(state, legal, agent.epsilon)
            
            if action is None:
                break
                
            next_state, reward, done, _ = env.step(action)
            next_legal = env.legal_actions()
            
            agent.update(state, action, reward, next_state, next_legal, done)
            state = next_state
            
        # Decay epsilon
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        
        if ep % 10000 == 0:
            print(f"Episode {ep}/{episodes} | Epsilon: {agent.epsilon:.3f} | Table Size: {len(agent.q_table)}")
            
    agent.save_q_table(f"q_table_{config}.pkl")
    print(f"Saved Q-table to q_table_{config}.pkl\n")
    return agent


if __name__ == '__main__':
    random.seed(42)
    # Required Baseline: Terminal rewards only, against RandomAgent
    train_agent(30000, "baseline", RandomAgent('O'), q_player='X')
    
    # Required Improved: Step penalty included, against NoisyHeuristicAgent
    train_agent(30000, "improved", NoisyHeuristicAgent('O'), q_player='X')