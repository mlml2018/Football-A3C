import torch
import os
from a3c import SharedAdam, ActorCritic
from football_game_ai_visualisation import Football_Game_Visualisation
import pygame

def load_model_checkpoint(filename='model_checkpoint.pth'):
    """
    Loads a saved model checkpoint
    
    Returns:
    - Loaded model
    - Loaded optimizer state
    - Hyperparameters
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")
    
    # Load the checkpoint
    checkpoint = torch.load(filename)
    
    # Recreate the model with saved hyperparameters
    model = ActorCritic(
        input_dims=checkpoint['hyperparameters']['input_dims'],
        n_discrete=checkpoint['hyperparameters']['n_discrete'],
        n_continuous=checkpoint['hyperparameters']['n_continuous'],
        gamma=checkpoint['hyperparameters']['gamma']
    )
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optional: Create an optimizer and load its state
    optimizer = SharedAdam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def create_env():
    # Reset Pygame state before creating a new environment
    pygame.quit()
    print("Creating environment...")
    return Football_Game_Visualisation()

def evaluate_model(checkpoint_path, num_episodes=10):
    """
    Load a saved model and evaluate its performance
    
    Args:
    - checkpoint_path: Path to the saved model checkpoint
    - num_episodes: Number of episodes to run for evaluation
    """
    # Load the model
    model, _ = load_model_checkpoint(checkpoint_path)
    
    # Create environment
    env = create_env()
    
    # Performance tracking
    performance_log = []
    
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Get action from the model
            action = model.get_action(observation)
            print(action)
            
            # Take step in environment
            observation_, reward, done = env.play_step(action)
            
            total_reward += reward
            observation = observation_
        
        performance_log.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    # Calculate and print performance statistics
    import numpy as np
    print("\nPerformance Summary:")
    print(f"Mean Reward: {np.mean(performance_log)}")
    print(f"Std Deviation: {np.std(performance_log)}")
    print(f"Best Reward: {np.max(performance_log)}")
    print(f"Worst Reward: {np.min(performance_log)}")
    
    return performance_log

if __name__ == "__main__":
    filename = 'football_a3c_model_param.pth'
    evaluate_model(filename)
