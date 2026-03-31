# Football AI

This project implements a simulated football game and an AI agent trained to play it using PPO.

## Project Structure

- `src/`: Contains all the source code for the game and the AI.
  - `football_game_human.py`: Play the game manually.
  - `football_game_ai.py` / `football_game_ai_visualisation.py`: The RL environment.
  - `train.py`: Central training script with multi-algorithm support.
  - `visualize.py`: Script to visualize a trained AI agent playing.
  - `agents/`: Contains the specific RL algorithm implementations (`ppo.py`, `dqn.py`, `a3c.py`).
- `models/`: Directory where trained PyTorch models are saved.
- `assets/`: Any images or assets used by the game.

## Requirements

The project requires Python 3. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Play the Game Manually
To play the game yourself as a human:
```bash
python src/football_game_human.py
```

### Train the AI
You can train the AI agent using different reinforcement learning algorithms. (Currently fully implemented: `ppo`. Stubs for `dqn`, `a3c`).
```bash
python src/train.py --algo ppo
```
This will train the agent and periodically save model checkpoints into the `models/` directory.

### Visualize Trained AI
To watch a trained agent play the game:
```bash
python src/visualize.py --algo ppo
```
*(Make sure you have a trained model in the `models/` directory before running the visualization).*
