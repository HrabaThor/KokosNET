from torch import nn
import numpy as np
from torch.distributions import Categorical, Normal

# Internal imports
from networkUtils import FlexibleDenseNet
from utils import to_tensor

# Some constants
GRAD_CLIP = 0.5         # Coefficient for gradient clipping (improves learning stability)


class ACCompact(nn.Module):
    def __init__(self, observation_space, action_space, hidden=[512, 512, 256],
                 policy=[32, 8], value=[32, 8], policy_a=nn.Softmax,
                 value_a=None, activations=nn.ReLU):
        super().__init__()
        # Initialize the common model
        self.model = FlexibleDenseNet(
            [observation_space] + hidden, activations=activations, last_a=activations
        )
        # Initialize policy submodel
        self.policy = FlexibleDenseNet(
            hidden[-1:] + policy + [action_space], activations=activations, last_a=policy_a
        )
        # Endpoint for values
        self.value = FlexibleDenseNet(
            hidden[-1:] + value + [1], activations=activations, last_a=value_a
        )
        
    def forward(self, x):
        # Apply common model
        common = self.model(x)
        # Policy
        p = self.policy(common)
        # Value
        v = self.value(common)
        return v, p

    def choose_actions(self, state, discrete, std):
        '''Choose next actions based on observation'''
        # Get actions from actor-critic neural network
        _, policies = self(to_tensor(np.array(state)))
        # Categorical for discreet env; Normal for continuous env
        if discrete:
            action_probabilities = Categorical(policies)
        else:
            action_probabilities = Normal(policies, std)
        # Sample from distributions and store the action
        action = action_probabilities.sample()
        # Reduce actions dimensions
        return action.reshape(-1,1)

    def learn(self, states, states_next, rewards, dones, actions, gamma, beta, optimizer, discrete, std):
        '''Single learning step'''
        # Adjust types and sizes
        states, states_next, rewards = to_tensor(states), to_tensor(states_next), to_tensor(rewards)
    
        # Inference neural network model
        values, policies = self(states)
        values_next, _ = self(states_next)
        
        # Flatten
        values, values_next, policies = values.flatten(), values_next.flatten(), policies.flatten()
        
        # Distribution of possible actions
        if discrete:
            distributions = Categorical(policies)
        else:
            distributions = Normal(policies, std)
        
        # Calculate loss
        policy_logs = distributions.log_prob(actions)
        entropies = distributions.entropy().mean()
        
        deltas = rewards + gamma * values_next * to_tensor(1 - dones.astype(bool)) - values
        actor_loss = - policy_logs * deltas - entropies * beta
        critic_loss = deltas ** 2
        total_loss = (actor_loss + critic_loss).sum()
        
        # Apply gradients
        optimizer.zero_grad()
        total_loss.backward()
        # Restrain gradients
        nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP)
        optimizer.step()