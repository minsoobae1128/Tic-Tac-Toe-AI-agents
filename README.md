# Tic-Tac-Toe-AI-agents
인공지능개론 과목 과제

# SWE3052-41 Homework 3: Reinforcement Learning (Q-Learning)

**Author:** 배민수

## Overview
This project implements a Tabular Q-Learning agent designed to learn and play 5x5 Tic-Tac-Toe (4-in-a-row to win). Due to the massive state space of a 5x5 board (approx. 3^25 combinations), the agent utilizes a dynamic, sparse dictionary-based Q-table to store and update state-action values organically as it encounters them. 

The project evaluates the agent under two distinct training configurations:
1. **Baseline Configuration:** Terminal-only rewards, trained against a purely random opponent.
2. **Improved Configuration:** Reward shaped (step-penalties), trained against a structured heuristic opponent.

## File Structure

* `agents.py` - Contains all agents (`RandomAgent`, `NoisyHeuristicAgent`, `AlphaBetaAgent` , 'QLearningAgent').
* `game.py` - Contains the `Board` class and the `Game` environment loop.
* `train.py` - The script to train both the baseline and improved configurations. Saves the resulting models as `.pkl` files.
* `evaluate.py` - The script to load the trained models, evaluate them in 0-exploration matches, and generate metrics.
* `q_table_baseline.pkl` - The saved Q-table memory dictionary trained against the Baseline configuration.
* `q_table_improved.pkl` - The saved Q-table memory dictionary trained against the Improved configuration.
* `results.csv` - Raw, row-by-row evaluation game logs containing match outcomes and move lengths.
* `README.md` - Setup instructions, dependencies, how to train, and how to evaluate.

## Dependencies and Setup
This project strictly adheres to the assignment's limitation on external libraries. All core logic is implemented using standard Python structures and the math/random utilities provided by `numpy`. 

To install the minimal required dependencies, run:
```bash
pip install -r requirements.txt

## How to run
Training the agents
py train.py

Evaluating the agents
py evaluate.py
