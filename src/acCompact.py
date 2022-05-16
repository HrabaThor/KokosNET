from torch import nn
import torch
import numpy as np
from torch.distributions import Categorical, Normal

# Internal imports
from networkUtils import FlexibleDenseNet, to_tensor

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
        self.losses = {
            'total': np.empty(0),
            'actor': np.empty(0),
            'critic': np.empty(0),
            'steps': np.empty(0)
        }
        
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
        values, policies = self(to_tensor(np.array(state)))
        # Categorical for discreet env; Normal for continuous env
        if discrete:
            action_dist = Categorical(policies)
        else:
            action_dist = Normal(policies, std)
        # Sample from distributions and store the action
        action = action_dist.sample()
        # Reduce actions dimensions
        return values.flatten(), action_dist, action.reshape(-1,1)


    def learn(self, optimizer, rewards, values, log_probs, states, positions, entropy, gamma, beta):
        '''Single learning step'''
        # Calculate the step
        step = rewards.shape[1]
        # Get the final value
        value, _ = self(to_tensor(states))
        discounts = [np.append([d], np.zeros(step - p)) for p, d in zip(positions, value.detach().numpy())]

        # Calculate discounted values
        for index, (position, reward) in enumerate(zip(positions, rewards)):
            # Reverse the rewards, we care the most about the last
            reward = reward[position:][::-1]
            # Calculate discounts
            for i, r in enumerate(reward, 1):
                discounts[index][i] = discounts[index][i-1] * gamma + r
            # Ignore approximated values from neural network
            discounts[index] = discounts[index][:-1][::-1]
 
        # TODO: Check if properly detached
        discounts = torch.FloatTensor(np.concatenate(discounts))
        values = torch.cat([ v[p:] for p, v in zip(positions, values)])
        log_probs = torch.cat([ l[p:] for p, l in zip(positions, log_probs)])
        advantage = discounts - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        total_loss = (actor_loss + critic_loss + beta * entropy).mean()
        print(total_loss)
        import pdb; pdb.set_trace()
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()


    # def learn(self, states, states_next, rewards, dones, actions, gamma, beta, optimizer, discrete, std):
    #     '''Single learning step'''
    #     # Adjust types and sizes
    #     states, states_next, rewards = to_tensor(states), to_tensor(states_next), to_tensor(rewards)
    
    #     # Inference neural network model
    #     values, policies = self(states)
    #     values_next, _ = self(states_next)
        
    #     # Flatten
    #     values, values_next, policies = values.flatten(), values_next.flatten(), policies.flatten()
        
    #     # Distribution of possible actions
    #     if discrete:
    #         distributions = Categorical(policies)
    #     else:
    #         distributions = Normal(policies, std)
        
    #     # Calculate loss
    #     policy_logs = distributions.log_prob(actions)
    #     entropies = distributions.entropy().mean()
        
    #     deltas = rewards + gamma * values_next * to_tensor(1 - dones.astype(bool)) - values
    #     actor_loss = - policy_logs.flatten() * deltas  - entropies * beta
    #     critic_loss = deltas ** 2
    #     total_loss = actor_loss + critic_loss
    #     self.losses['total'] = np.append(self.losses['total'], total_loss.detach().numpy())
    #     self.losses['critic'] = np.append(self.losses['critic'], critic_loss.detach().numpy())
    #     self.losses['actor'] = np.append(self.losses['actor'], actor_loss.detach().numpy())
    #     # Apply gradients
    #     optimizer.zero_grad()
    #     total_loss.mean().backward()
    #     # Restrain gradients
    #     nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP)
    #     optimizer.step()
