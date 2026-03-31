import torch
import os
from ppo import PPOAgent, RunningMeanStd
import numpy as np
import time

# Use the fixed visualisation environment
from football_game_ai_visualisation import Football_Game_Visualisation

def evaluate_model(checkpoint_path, num_episodes=5):
    """
    Load a saved model and evaluate its performance with rendering
    """
    if not os.path.exists(checkpoint_path):
        print(f"Model not found at {checkpoint_path}")
        return
        
    # Create environment
    env = Football_Game_Visualisation()
    
    # Init agent and load
    agent = PPOAgent()
    obs_rms = RunningMeanStd(shape=(10,))
    agent.load(checkpoint_path, obs_rms)
    agent.network.eval()
    
    for episode in range(num_episodes):
        raw_observation = env.reset()
        done = False
        total_reward = 0
        
        # We need a small delay in visualisation so humans can see it
        # Actually clock.tick(FPS) in play_step already limits it.
        while not done:
            # Apply normalisation without updating running stats (evaluation mode)
            norm_obs = np.clip((raw_observation - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                # Sample from the distribution instead of argmax to preserve exploratory policies
                p1_logits, p2_logits, pass_logits, _ = agent.network(obs_tensor)
                
                from torch.distributions import Categorical
                a_p1 = Categorical(logits=p1_logits).sample().squeeze(0).float()
                a_p2 = Categorical(logits=p2_logits).sample().squeeze(0).float()
                a_pass = Categorical(logits=pass_logits).sample().squeeze(0).float()
                
                action = torch.tensor([a_p1, a_p2, a_pass]).cpu().numpy()
            
            raw_observation, reward, done = env.play_step(action)
            total_reward += reward
            
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == "__main__":
    model_path = '../models/football_ppo_model.pth'
    evaluate_model(model_path)
