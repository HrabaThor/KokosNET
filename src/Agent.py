# Main module implementing an Agent capable of learning and performing actions
# in pytorch gym environment
import torch
import torch.nn as nn
from torch import optim
from gym import make
import numpy as np
from agentHandle import AgentHandle
import yaml
from networkUtils import to_tensor

# Import neural network module
from acCompact import ACCompact

# Default configuration of neural network
CFG = {
    'arch': "ACCompact",      # Class of neural network to be used
    'h': [512,256],             # Hidden layers
    'a': nn.ReLU,               # Activations through the net (last activations for policy and values excluded)
    'p': [],                    # Policy layers
    'pa': nn.Softmax,           # Last policy network module activation
    'v': [],                    # Value layers
    'va': None                  # Last value network module activation
}

class Agent:
    '''
    Implementation of agent performing and training in gym environment.

    Some points:
        This agent uses pseudo-parallel training (sequentially training on multiple envs)
            in order to decorrelate the training data
    
    Provides:
        __init__:   Initialize the agent, provide env id and it's "dicreteness",
                    sampling std for actions and neural network config
        evaluate:   Evaluate agent's performance in it's or provided envirnoment
        train:      Train the agent's neural network - neural network should provide learn method
        net_init:   Initialize neural network from provided configuration
    '''
    def __init__(self, environment="CartPole-v1", discrete=True, sample_std=0.04, net_cfg=CFG):
        '''
        Initialize agent and it's neural network from net_cfg config dictionary

        Arguments:
            envirnoment:    Pytorch gym environment id
            discrete:       Whether provided envirnoment is discrete
            sample_std:     Standard deviation when sampling actions from normal distribution
            net_cfg:        Config dictionary to determine the neural network used
                            (for different neural networks, different keys and values. BEWARE!)
        '''
        self.lr = 0                                       # Place holder for latter use
        self.env_id = environment                         # Keep track of env required
        self.discrete = discrete                          # Discrete env?
        self.best, self.max_reward = None, float('-inf')  # Path to best model and it's reward
        self.sample_std = sample_std                      # STD for sampling action from policy distribution
        self.history = {'rewards': np.empty(0)}           # Variable for storing some histarical values of self
        self.net_config = net_cfg                         # Save the network configuration for init_network method

        # Make default envirnoment to be used for evaluating purposes (need batch of envs for training)
        self.env = make(environment)
        
        # Size of environment state vector
        self.observation_space = self.env.observation_space.shape[0]
        # Size of environment action vector
        self.n_actions = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        # Initialize it's neural network
        self.init_network()
    
    def load_config(self):
        '''Loads network configuration from file in self.net_config'''
        with open(self.net_config, 'r') as config:
            cfg = yaml.load(config, yaml.FullLoader)
        # Cycle over it and convert text to some actual objects
        for key, value in cfg.items():
            if isinstance(value, str) and value.startswith("nn."):
                cfg[key] = getattr(torch, value)
        self.net_config = cfg

    def init_network(self):
        '''Initialize neural network from configuration in self.net_config (dictionary or YAML file with config)'''
        if not isinstance(self.net_config, dict):
            self.load_config()
        # TODO: Remove this line of code, this is just to make it easier to swithch 
        # from continuous to discrete without changing the network configuration
        if self.discrete:    # If the environment is descrete, policy acativation function is probably Softmax
            self.net_config['pa'] = nn.Softmax

        # Initialize the neural network based on cfg arugment
        if self.net_config['arch'] == "ACCompact":
            self.net = ACCompact(self.observation_space, self.n_actions,
                                   activations=self.net_config['a'], policy_a=self.net_config['pa'], value_a=self.net_config['va'],
                                   policy=self.net_config['p'], value=self.net_config['v'], hidden=self.net_config['h'])

    def evaluate(self, epochs=1, env=None):
        '''Evaluate agent in environment'''
        # Choose environment based on our requirement to see agent in action
        if not env:
            env = self.env
        rewards_history = []      # Track all rewards
        for _ in range(epochs):   # Main loop
            state = env.reset()
            done = False
            rewards = 0           # Keep track of rewards
            while not done:       # Single game loop
                action = self.net.choose_actions(state[np.newaxis, :], self.discrete, 0)[0]
                state_next, reward, done, _ = env.step(action.item())
                rewards += reward
                state = state_next
            rewards_history.append(rewards)
        env.close()
        return rewards_history


    # def train(self, epochs=1000, lr=0.0003, gamma=0.99, beta=0.001, gyms=16,
    #           path="models", checkpoint=50):
    #     self.lr = lr
    #     # Initialize optimizer
    #     optimizer = optim.Adam(self.net.parameters(), lr)
    #     # Cache to keep track of learning, TODO: Improve cache
    #     cache = np.empty(0)

    #     # Create all environments
    #     envs = np.array([make(self.env_id) for _ in range(gyms)])
    #     states = np.stack([env.reset() for env in envs])

    #     # Main training loop
    #     epoch = 0
    #     entropies = to_tensor(np.zeros(gyms))
    #     rewards = np.array([np.empty(0)]*gyms)
    #     log_probs = torch.tensor([np.empty(0)]*gyms)
    #     values = torch.tensor([np.empty(0)]*gyms)
    #     runs_from = np.zeros(gyms).astype(int)
    #     while epoch < epochs:
    #         # Pseudo-parallel learning (sequentially learning on multiple envs)
    #         value, action_dist, action = self.net.choose_actions(
    #             states, self.discrete, self.sample_std)  # Initial env evaluation with action selection and approximated Qvalues
    #         action = action.reshape((gyms))

    #         states_next, reward, dones, _ = np.array(       # Apply actions through envs
    #             [np.array(env.step(action.item())) for action, env in zip(action, envs)]).transpose()
    #         # Some datatype adjustements
    #         dones = dones.astype(bool)
    #         states_next = np.stack(states_next)

    #         # TODO: Waht is this? log_prob = torch.log(policy_dist.squeeze(0)[action])
    #         log_prob = action_dist.log_prob(action)
    #         # TODO: entropy = -np.sum(np.mean(dist) * np.log(dist))
    #         entropies += action_dist.entropy().mean()
    #         # Apply some neccessary formatting
    #         value = value.reshape((gyms, 1))
    #         reward = reward.reshape((gyms, 1))
    #         log_prob = log_prob.reshape((gyms, 1))
    #         # Store them for later use
    #         values = torch.hstack((values, value))
    #         rewards = np.hstack((rewards, reward))
    #         log_probs = torch.hstack((log_probs, log_prob))

    #         # optimize the actor critic network
    #         self.net.learn(optimizer, rewards, values, log_probs, states, runs_from, entropies, gamma, beta)
            
    #         # Increment by number of envs done (finished)
    #         epoch += sum(dones)
            
    #         ########## Choose next states or if done - reset the env and begin once again ########## MBY NUMPY
    #         if dones.any():
    #             # Get cumulative rewards of finished agents
    #             rewards_done = np.array([r[i:].sum() for i, r in zip(runs_from[dones], rewards[dones])])
    #             # Write rewards in history
    #             self.history['rewards'] = np.append(self.history['rewards'], rewards_done)
    #             cache = np.append(cache, rewards_done)

    #             # keep track of best performing agent
    #             r = rewards_done.mean() 
    #             if r >= self.max_reward:  # If best reward, save the agent
    #                 self.max_reward = r   # Get new reward and save weights with path
    #                 self.best = AgentHandle.pickle(self, filename=f"e{epoch}_r{r}", path=path)
            

    #             # Keep track of trajectories
    #             runs_from[dones] = rewards.shape[1]
    #             # Remove obsolete information
    #             min_id = runs_from.min()
    #             rewards, log_probs, values = rewards[min_id:], log_probs[min_id:], values[min_id:]
    #             # Update states by resetting envs only when agent finished (done)
    #             states[dones] = np.array([env_.reset() for env_ in envs[dones]])
    #         # Propagate all next states for env where agent did not finish
    #         states[np.logical_not(dones)] = states_next[np.logical_not(dones)]
            
    #         # Write out the progress
    #         if len(cache) >= checkpoint:
    #                     print("Epoch: {}\tCumulative reward: {:.2f}".format(epoch, cache.mean()))
    #                     cache = cache[checkpoint-1:]

    #     # Close all environments
    #     _ = [env.close() for env in envs]