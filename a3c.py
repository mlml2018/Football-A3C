import torch
from torch import nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import VonMises, Bernoulli, Categorical
from football_game_ai import Football_Game
import pygame
import os
import numpy as np

def save_model_checkpoint(model, optimiser, filename='model_checkpoint.pth'):
    """
    Saves the model weights, optimizer state, and additional metadata
    
    Args:
    - model: The neural network model
    - optimiser: The optimiser used during training
    - filename: Name of the checkpoint file
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'hyperparameters': {
            'input_dims': getattr(model, 'input_dims', None),
            'n_discrete': getattr(model, 'n_discrete', None),
            'n_continuous': getattr(model, 'n_continuous', None),
            'gamma': getattr(model, 'gamma', 0.99)
        }
    }
    
    torch.save(checkpoint, filename)
    print(f"Model checkpoint saved to {filename}")

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

def continue_training(checkpoint_path):
    try:
        # Load the existing model and optimizer
        model, optimiser = load_model_checkpoint(checkpoint_path)
        
        # Reset or modify training configuration as needed
        CONFIG = {
            'lr': 1e-3,
            'n_discrete': model.n_discrete,
            'n_continuous': model.n_continuous,
            'input_dims': model.input_dims,
            'N_GAMES': 500000,
            'T_MAX': 500,
            'gamma': model.gamma,
            'num_workers': min(8,mp.cpu_count())
        }
        global_ep = mp.Value('i', 0)
        model.share_memory()
        # Proceed with training using the loaded model
        workers = [
            Worker(
                global_actor_critic=model,
                optimiser=optimiser,
                input_dims=CONFIG['input_dims'],
                n_discrete=CONFIG['n_discrete'],
                n_continuous=CONFIG['n_continuous'],
                N_GAMES=CONFIG['N_GAMES'],
                T_MAX=CONFIG['T_MAX'],
                gamma=CONFIG['gamma'],
                lr=CONFIG['lr'],
                name=i,
                global_ep_idx=global_ep,
                env=create_env()
            ) for i in range(CONFIG['num_workers'])
        ]
        
        [w.start() for w in workers]
        [w.join() for w in workers]

        # Save final model checkpoint
        save_model_checkpoint(model, optimiser, 'final_model_checkpoint.pth')

    except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

# Adam optimiser for multiprocessing RL algorithms
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # State initialisation
        for group in self.param_groups:
            for p in group['params']:

                # for each p, initialise optimiser state with same shape as parameter
                state = self.state[p]
                # Initialise step with zero tensor
                state['step'] = torch.tensor(0,dtype=torch.long)
                # exponential moving average of gradients
                state['exp_avg']=  torch.zeros_like(p.data)
                # exponential moving average of squared gradients
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Share in memory
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# Defines neural network
# Architecture predicts two outputs: policy (pi) and value (v)
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_discrete, n_continuous, gamma=0.99):
        '''
        input_dims : dimensions of input state
        n_actions: number of actions
        gamma : discount factor for future rewards
        '''
        # Unpack input dimensions if it's a list
        if isinstance(input_dims, list):
            input_dim = input_dims[0]  # Take the first element if it's a list
        else:
            input_dim = input_dims

        super(ActorCritic, self).__init__()

        # For epsilon greedy method
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.99995  # Decay rate for epsilon
        self.epsilon_min = 0.1 #0.01  # Minimum exploration probability

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.input_dims = input_dims
        self.gamma = gamma
        self.n_discrete = n_discrete
        self.n_continuous = n_continuous

        self.pi_continuous_mu = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, n_continuous)
        )
        self.pi_continuous_kappa = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, n_continuous)
        )
        self.pi_discrete = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.v = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialise weights and biases, set to normal distribution
        set_init([self.feature_extractor[0], self.feature_extractor[2],
                  self.pi_continuous_mu[0], self.pi_continuous_mu[2],
                  self.pi_continuous_kappa[0], self.pi_continuous_kappa[2],
                  self.pi_discrete[0], self.pi_discrete[2],
                  self.v[0], self.v[2]])
        # store s,a,r during an episode - initialise to empty list
        self.rewards = []
        self.actions = []
        self.states = []
    
    def decay_epsilon(self, episode):
        """
        Decay epsilon over episodes
        
        Args:
        - episode: Current episode number
        """
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon * (self.epsilon_decay ** episode)
        )
        return self.epsilon

    def forward(self, state):
        # Ensure state is a 2D tensor
        if state.dim() == 1:
            state = state.unsqueeze(0)
        feature = self.feature_extractor(state)

        # Discrete action logits
        pi_discrete_logits = self.pi_discrete(feature)

        # Continuous action parameters for von Mises (low kappa values - more uniform)
        pi_continuous_mu = 180 * F.tanh(self.pi_continuous_mu(feature))
        pi_continuous_kappa = F.softplus(self.pi_continuous_kappa(feature)) + 0.1 # ensure kappa is greater than zero
        values= self.v(feature)

        return pi_discrete_logits, pi_continuous_mu, pi_continuous_kappa, values
    
    # stores s,a,r for each step in agent's trajectory
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    # clears s,a,r (for next episode)
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    # G_t^(n) = R_(t+1) + ... gamma^(n-1) * R_(n+t) + gamma^n * V(s_(n+t)
    def calc_R(self, done):
        '''
        returns list of G_t^(n) at each time step
        done: terminal flag
        '''
        states = torch.tensor(self.states, dtype=torch.float)
        _, _, _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        episode_return = []
        # loop through self.rewards in reverse direction
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            episode_return.append(R)
        episode_return.reverse()
        episode_return = torch.tensor(episode_return, dtype = torch.float)

        return episode_return
    

    def calc_loss(self, done):
        states = torch.tensor(self.states, dtype=torch.float)
        discrete_actions = torch.stack([
            torch.tensor(action[:self.n_discrete], dtype=torch.float)
            for action in self.actions
        ])
        continuous_actions = torch.stack([
            torch.tensor(action[self.n_discrete:], dtype=torch.float)
            for action in self.actions
        ])

        returns = self.calc_R(done)

        pi_discrete_logits, mu, kappa, values = self.forward(states)
        values = values.squeeze() # ensure dimensions matches with returns
        
        # Loss function of critic (target(~G(t))-V)**2
        
        critic_loss = F.mse_loss(returns, values)
        # Loss function of actor
        '''
        loss function: (G-V)*del(log(pi(a|s)))
        del is not implemented in the code below since the gradient of the log_probs
        term is taken with respect to the model parameters when loss.backward() is called
        '''
        # Discrete loss
        #print("pi_discrete_logits shape:", pi_discrete_logits.shape)
        #print("pi_discrete_logits:", pi_discrete_logits)
        probs = F.softmax(pi_discrete_logits, dim=1)
        dist = Categorical(probs=probs)
        discrete_log_probs = dist.log_prob(discrete_actions) # log(pi(a|s))

        # Continuous loss
        # Ensure mu and kappa are 1D and have no gradient
        mu = mu.detach().squeeze(0)
        kappa = kappa.detach().squeeze(0)
        mu = torch.clamp(mu, min=-180, max=180)
        kappa = torch.clamp(kappa, min=0.1, max=100.0)
        m = VonMises(mu, kappa)
        continuous_log_probs = m.log_prob(continuous_actions)

        # Compute actor loss       
        actor_loss = -torch.mean((discrete_log_probs.sum(dim=1)+continuous_log_probs.sum(dim=1))*(returns-values)) # negative sign since optimisers minimise loss by default but this is gradient ascent
        # have to sum the loss because of the way that backpropagation is handled by torch
        total_loss = critic_loss + actor_loss #- 0.01 * (entropy_discrete + entropy_continuous)
        
        return total_loss
    
    def get_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float)

        # Ensure it's a 2D tensor
        if state.dim() == 1:
            state = state.unsqueeze(0)

        pi_discrete_logits, mu, kappa, _ = self.forward(state)

        # Discrete space
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random action selection
            discrete_action = torch.tensor(
                np.random.randint(0, 2), 
                dtype=torch.float
            ).unsqueeze(0)
        else:
            # Greedy action selection
            disc_probs = F.softmax(pi_discrete_logits, dim=1)
            disc = Categorical(probs=disc_probs)
            discrete_action = disc.sample().float()

        # Continuous space
        # Use .squeeze(0) to ensure mu and sigma are 1D tensors matching n_continuous
        mu = mu.squeeze(0)
        kappa = kappa.squeeze(0)
        cont = VonMises(mu, kappa)
        continuous_action_rad = cont.sample()
        continuous_action_deg = torch.rad2deg(continuous_action_rad)

        combined_action = torch.cat((discrete_action, continuous_action_deg),dim = 0) # shape: [n_discrete+n_continuous]
    
        return combined_action.numpy()

class Worker(mp.Process):
    def __init__(self, global_actor_critic, optimiser, input_dims, n_discrete, n_continuous, N_GAMES, T_MAX,
               gamma, lr, name, global_ep_idx, env):
        super(Worker, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_discrete, n_continuous, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%01i' % name # assign unique name to each worker for identification
        self.episode_idx = global_ep_idx
        self.global_lock = mp.Lock() # Define global Lock shared across workers
        self.env = env
        self.optimiser = optimiser
        self.N_GAMES = N_GAMES
        self.T_MAX = T_MAX
    
    def run(self):
        t_step = 1
        
        while self.episode_idx.value < self.N_GAMES:
            print(f"{self.name}: Starting episode {self.episode_idx.value}")
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            # Add a timeout mechanism
            max_steps = 500  # Prevent infinite loops
            current_step = 0
            while not done and current_step < max_steps:
                action = self.local_actor_critic.get_action(observation)
                observation_, reward, done = self.env.play_step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % self.T_MAX == 0 or done: # synchronise with global model
                    
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimiser.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimiser.step()
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                current_step += 1
                t_step += 1
                observation = observation_ # move to next state
            # .get_lock() ensures thread-safe incrementing of shared episode_idx across all workers
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
                print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)
        
def create_env():
    # Reset Pygame state before creating a new environment
    pygame.quit()
    print("Creating environment...")
    return Football_Game()

if __name__ == '__main__':
    #load_model_path = None
    load_model_path = 'football_a3c_model_param.pth'
    if load_model_path:
        print("continue training...")
        continue_training(load_model_path)
    else:
        try:
            mp.set_start_method('spawn')

            # Hyperparameters
            CONFIG = {
                'lr': 1e-3,
                'n_discrete': 1,
                'n_continuous': 3,
                'input_dims': [10],
                'N_GAMES': 5000, # 2000
                'T_MAX': 500,
                'gamma': 0.99,
                'num_workers': min(8,mp.cpu_count())
            }

            global_actor_critic = ActorCritic(
                CONFIG['input_dims'], 
                CONFIG['n_discrete'], 
                CONFIG['n_continuous']
            )
            global_actor_critic.share_memory()
            
            optim = SharedAdam(
                global_actor_critic.parameters(), 
                lr=CONFIG['lr'], 
                betas=(0.92, 0.999)
            )
            
            global_ep = mp.Value('i', 0)
            
            workers = [
                Worker(
                    global_actor_critic,
                    optim,
                    CONFIG['input_dims'],
                    CONFIG['n_discrete'],
                    CONFIG['n_continuous'],
                    CONFIG['N_GAMES'],
                    CONFIG['T_MAX'],
                    gamma=CONFIG['gamma'],
                    lr=CONFIG['lr'],
                    name=i,
                    global_ep_idx=global_ep,
                    env=create_env()
                ) for i in range(CONFIG['num_workers'])
            ]
            
            [w.start() for w in workers]
            [w.join() for w in workers]

            # Save final model checkpoint
            save_model_checkpoint(global_actor_critic, optim, 'football_a3c_model_param.pth')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
