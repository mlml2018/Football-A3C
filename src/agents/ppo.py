import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import os

class RunningMeanStd:
    # Dynamically tracks the mean and variance of the observations
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim=10):
        super(PPOActorCritic, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        self.v = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.pi_p1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )
        self.pi_p2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )
        self.pi_pass = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Value head should have smaller weights initially
        nn.init.orthogonal_(self.v[-1].weight, gain=1.0)
        # Action heads MUST have near-zero weights so the initial distribution is uniform (entropy preservation)
        nn.init.orthogonal_(self.pi_p1[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.pi_p2[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.pi_pass[-1].weight, gain=0.01)
        
    def forward(self, state):
        features = self.feature_extractor(state)
        value = self.v(features)
        
        p1_logits = self.pi_p1(features)
        p2_logits = self.pi_p2(features)
        pass_logits = self.pi_pass(features)
        
        return p1_logits, p2_logits, pass_logits, value
        
    def get_action_and_value(self, state, action=None):
        p1_logits, p2_logits, pass_logits, value = self(state)
        
        dist_p1 = Categorical(logits=p1_logits)
        dist_p2 = Categorical(logits=p2_logits)
        dist_pass = Categorical(logits=pass_logits)
        
        if action is None:
            a_p1 = dist_p1.sample()
            a_p2 = dist_p2.sample()
            a_pass = dist_pass.sample()
            
            sampled_action = torch.stack([a_p1, a_p2, a_pass], dim=-1)
            a_p1_eval = a_p1
            a_p2_eval = a_p2
            a_pass_eval = a_pass
        else:
            a_p1_eval = action[:, 0].long()
            a_p2_eval = action[:, 1].long()
            a_pass_eval = action[:, 2].long()
            
        log_prob = dist_p1.log_prob(a_p1_eval) + dist_p2.log_prob(a_p2_eval) + dist_pass.log_prob(a_pass_eval)
        entropy = dist_p1.entropy() + dist_p2.entropy() + dist_pass.entropy()
        
        if action is None:
            return sampled_action.float(), log_prob, entropy, value
        else:
            return log_prob, entropy, value

class PPOAgent:
    def __init__(self, input_dim=10, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, c1=0.5, c2=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = PPOActorCritic(input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.c1 = c1 # Value loss coefficient
        self.c2 = c2 # Entropy coefficient
        
    def update(self, rollouts):
        states, actions, logprobs, rewards, dones, values = rollouts
        
        # Calculate advantages using GAE
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = 0 # Assume we end on a terminal or bootstrap
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_states = states.view(-1, states.shape[-1])
        b_actions = actions.view(-1, actions.shape[-1])
        b_logprobs = logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = values.view(-1)

        # Optimize policy and value networks (epochs)
        epochs = 4
        batch_size = 64
        b_inds = np.arange(len(b_states))
        
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_states), batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]
                
                # NORMALIZE ADVANTAGES PER MINIBATCH (Critical PPO Fix)
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                newlogprob, entropy, newvalue = self.network.get_action_and_value(b_states[mb_inds], b_actions[mb_inds])
                newvalue = newvalue.view(-1)
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = F.mse_loss(newvalue, b_returns[mb_inds])
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.c2 * entropy_loss + self.c1 * v_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

    def save(self, path, obs_rms=None):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_rms_mean': obs_rms.mean if obs_rms else None,
            'obs_rms_var': obs_rms.var if obs_rms else None,
        }, path)

    def load(self, path, obs_rms=None):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if obs_rms and 'obs_rms_mean' in checkpoint and checkpoint['obs_rms_mean'] is not None:
            obs_rms.mean = checkpoint['obs_rms_mean']
            obs_rms.var = checkpoint['obs_rms_var']

def normalize_obs(obs, obs_rms):
    obs_rms.update(np.array([obs]))
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

def train_ppo(env):
    agent = PPOAgent()
    
    num_updates = 100000
    steps_per_update = 2048
    
    global_step = 0
    episode_score = 0
    episode_count = 0
    
    # Storage
    states = torch.zeros((steps_per_update, 10)).to(agent.device)
    actions = torch.zeros((steps_per_update, 3)).to(agent.device)
    logprobs = torch.zeros((steps_per_update,)).to(agent.device)
    rewards = torch.zeros((steps_per_update,)).to(agent.device)
    dones = torch.zeros((steps_per_update,)).to(agent.device)
    values = torch.zeros((steps_per_update,)).to(agent.device)
    
    obs_rms = RunningMeanStd(shape=(10,))
    raw_obs = env.reset()
    norm_obs = normalize_obs(raw_obs, obs_rms)
    obs = torch.tensor(norm_obs, dtype=torch.float32).to(agent.device)
    
    try:
        for update in range(1, num_updates + 1):
            
            for step in range(steps_per_update):
                global_step += 1
                
                with torch.no_grad():
                    action, logprob, _, value = agent.network.get_action_and_value(obs.unsqueeze(0))
                
                states[step] = obs
                actions[step] = action.squeeze(0)
                logprobs[step] = logprob.squeeze(0)
                values[step] = value.squeeze(0)
                
                # Step env
                action_np = action.squeeze(0).cpu().numpy()
                
                raw_next_obs, reward, done = env.play_step(action_np)
                norm_next_obs = normalize_obs(raw_next_obs, obs_rms)
                next_obs = torch.tensor(norm_next_obs, dtype=torch.float32).to(agent.device)
                
                # Scale reward to prevent Value Network gradient explosion
                rewards[step] = reward / 10.0
                dones[step] = done
                
                obs = next_obs
                episode_score += reward
                
                if done:
                    episode_count += 1
                    print(f"Episode {episode_count}, Reward: {episode_score:.2f}, Global Step: {global_step}")
                    
                    if episode_count % 10 == 0:
                        # Print the actual distinct discrete choices of the last 5 steps to show it's localized
                        recent_max = min(step + 1, 5)
                        if recent_max > 0:
                            recent_actions = actions[step - recent_max + 1:step + 1].cpu().numpy()
                            print(f"  -> Last {recent_max} P1 Actions: {recent_actions[:, 0].flatten()}")
                            print(f"  -> Last {recent_max} Pass Dirs : {recent_actions[:, 2].flatten()}")
                            
                    raw_obs = env.reset()
                    norm_obs = normalize_obs(raw_obs, obs_rms)
                    obs = torch.tensor(norm_obs, dtype=torch.float32).to(agent.device)
                    episode_score = 0
                    
            print(f"Updating PPO model at step {global_step} (Update {update})...")
            agent.update((states, actions, logprobs, rewards, dones, values))
            
            # Save checkpoint regularly
            if update % 5 == 0:
                os.makedirs('models', exist_ok=True)
                agent.save('models/football_ppo_model.pth', obs_rms)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving current model state...")
        os.makedirs('models', exist_ok=True)
        agent.save('models/football_ppo_model.pth', obs_rms)
        print("Model saved successfully. Exiting.")
