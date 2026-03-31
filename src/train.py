import argparse
import sys
import os

# Add the src folder to Python path to allow agents to import football_game_ai
# when running `python src/train.py` from root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from football_game_ai import Football_Game
from agents.ppo import train_ppo
from agents.dqn import train_dqn
from agents.a3c import train_a3c

def main():
    parser = argparse.ArgumentParser(description="Football AI Training Entry Point")
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'dqn', 'a3c'], 
                        help='The reinforcement learning algorithm to use')
    args = parser.parse_args()

    env = Football_Game()
    
    if args.algo == 'ppo':
        print("Starting PPO training...")
        train_ppo(env)
    elif args.algo == 'dqn':
        print("Starting DQN training...")
        train_dqn(env)
    elif args.algo == 'a3c':
        print("Starting A3C training...")
        train_a3c(env)

if __name__ == '__main__':
    main()
