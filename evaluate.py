import random
import csv
from game import Board
from agents import QLearningAgent, RandomAgent, NoisyHeuristicAgent, AlphaBetaAgent
from train import TicTacToeRLEnv

def print_q_value_heatmap(board_obj, q_values, chosen_action):
    """Draws a text-based grid showing X, O, and the Q-values for empty spaces."""
    print("\n" + "="*45)
    print("      Q-VALUE HEATMAP & BOARD STATE")
    print("="*45)
    
    for r in range(5):
        row_str = []
        for c in range(5):
            cell = board_obj.grid[r][c]
            if cell is not None:
                row_str.append(f"   {cell}   ")
            else:
                # Empty space: display the Q-value
                q = q_values.get((r, c), 0.0)
                if (r, c) == chosen_action:
                    row_str.append(f"[{q:>+5.2f}]")  # Highlight chosen action
                else:
                    row_str.append(f" {q:>+5.2f} ")
                    
        print(" | ".join(row_str))
        if r < 4:
            print("-" * 45)
    print("="*45)

def run_visual_evaluation(q_table_path, opponent_class):
    """Runs a single game step-by-step for the visualization requirement."""
    print(f"\n>>> STARTING VISUAL EVALUATION DEMO ({q_table_path}) <<<")
    agent = QLearningAgent(player='X')
    
    try:
        agent.load_q_table(q_table_path)
    except FileNotFoundError:
        print(f"Error: {q_table_path} not found. Please run train.py first.")
        return

    board = Board()
    agent_player = 'X'
    opponent = opponent_class(player='O')
    done = False
    moves = []
    
    # Explicitly track turns
    current_turn = 'X'

    while not done:
        state = tuple(tuple(row) for row in board.grid)
        legal = board.get_legal_moves()

        # --- Q-LEARNING AGENT TURN ---
        if current_turn == agent_player:
            q_values = agent.q_table.get(state, {m: 0.0 for m in legal})
            
            # Select best action (Evaluation mode: epsilon=0)
            best_val = max(q_values.values()) if q_values else 0.0
            best_moves = [m for m in legal if q_values.get(m, 0.0) == best_val]
            chosen_action = random.choice(best_moves) if best_moves else random.choice(legal)

            # Print heatmap and chosen action
            print_q_value_heatmap(board, q_values, chosen_action)
            print(f"Player X (Q-Agent) plays {chosen_action} with Q-Value: {q_values.get(chosen_action, 0.0):.3f}")
            
            board.make_move(chosen_action[0], chosen_action[1], agent_player)
            moves.append(('X', chosen_action[0], chosen_action[1]))
            current_turn = 'O'
            
        # --- OPPONENT TURN ---
        else:
            action = opponent.choose_move(board)
            if action:
                print(f"\nPlayer O (Opponent) plays {action}")
                board.make_move(action[0], action[1], 'O')
                moves.append(('O', action[0], action[1]))
            current_turn = 'X'
            
            # Print standard board after opponent move
            print("-" * 21)
            for row in board.grid:
                print(" | ".join([cell if cell else " " for cell in row]))
            print("-" * 21)

        winner = board.check_winner()
        if winner or board.is_draw():
            done = True

    # Final Result and Summary Statistics
    print("\n" + "*"*30)
    print("      FINAL RESULT SUMMARY")
    print("*"*30)
    for row in board.grid:
        print(" | ".join([cell if cell else " " for cell in row]))
        
    if winner:
        print(f"\nResult: Player {winner} Wins!")
    else:
        print("\nResult: Draw!")
        
    print(f"Total Moves Played: {len(moves)}")
    print("*"*30 + "\n")

def evaluate_model(q_model_path, opponent_class, opponent_name, config, n_games=100):
    agent = QLearningAgent(player='X')
    agent.load_q_table(q_model_path)
    
    results = []
    wins, losses, draws = 0, 0, 0
    total_reward = 0.0
    
    print(f"Evaluating {config.upper()} Agent vs {opponent_name} ({n_games} games)...")
    
    for i in range(n_games):
        random.seed(i)
        
        if opponent_name == "AlphaBetaAgent":
            opp_agent = opponent_class(player='O', depth=2) 
        else:
            opp_agent = opponent_class(player='O')
            
        env = TicTacToeRLEnv(q_player='X', opponent_agent=opp_agent, config=config)
        
        q_starts = True if i < (n_games // 2) else False
        state = env.reset(q_starts=q_starts)
        done = False
        moves_count = 0
        ep_reward = 0.0
        
        while not done:
            legal = env.legal_actions()
            action = agent.select_action(state, legal, epsilon=0.0)
            
            if not action:
                break
                
            state, reward, done, info = env.step(action)
            moves_count += 1
            ep_reward += reward
            
        winner = info.get('winner')
        if winner == 'X':
            outcome = 'Win'
            wins += 1
        elif winner == 'O':
            outcome = 'Loss'
            losses += 1
        else:
            outcome = 'Draw'
            draws += 1
            
        total_reward += ep_reward
        results.append({
            'Config': config,
            'Opponent': opponent_name,
            'Game_ID': i,
            'Q_Starts': q_starts,
            'Outcome': outcome,
            'Moves': moves_count,
            'Total_Reward': ep_reward
        })
        
    avg_reward = total_reward / n_games
    print(f"  -> Wins: {wins} | Losses: {losses} | Draws: {draws} | Avg Reward: {avg_reward:.2f}\n")
    return results

if __name__ == '__main__':
    # 1. Run the Single Visual Evaluation Game (satisfies Section 6)
    # This will print the board state step-by-step to the terminal
    random.seed(42)
    run_visual_evaluation("q_table_improved.pkl", NoisyHeuristicAgent)
    
    # 2. Run the Mass Evaluations and generate CSV
    all_results = []
    
    # Evaluate Baseline Model
    all_results.extend(evaluate_model("q_table_baseline.pkl", RandomAgent, "RandomAgent", "baseline", 100))
    all_results.extend(evaluate_model("q_table_baseline.pkl", NoisyHeuristicAgent, "NoisyHeuristicAgent", "baseline", 100))
    all_results.extend(evaluate_model("q_table_baseline.pkl", AlphaBetaAgent, "AlphaBetaAgent", "baseline", 20))
    
    # Evaluate Improved Model
    all_results.extend(evaluate_model("q_table_improved.pkl", RandomAgent, "RandomAgent", "improved", 100))
    all_results.extend(evaluate_model("q_table_improved.pkl", NoisyHeuristicAgent, "NoisyHeuristicAgent", "improved", 100))
    all_results.extend(evaluate_model("q_table_improved.pkl", AlphaBetaAgent, "AlphaBetaAgent", "improved", 20))
    
    with open('results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Config', 'Opponent', 'Game_ID', 'Q_Starts', 'Outcome', 'Moves', 'Total_Reward'])
        writer.writeheader()
        writer.writerows(all_results)
        
    print("Metrics written successfully to results.csv.")