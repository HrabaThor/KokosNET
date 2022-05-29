import numpy as np
import torch
import os
from nets import Critic, Actor
import gym
from gym.wrappers import Monitor
from tqdm import tqdm
from time import sleep
from buffer import ReplayBuffer
from torch.nn.functional import mse_loss
#import torchviz
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns

class Agent:
    def __init__(self, env_id='Pendulum-v1', noise=0.1, buffer=1000000, warmup=100000,
                 model_dir="models", lra=0.001, lrc=0.001, tau=0.005, gamma=0.99,
                 c_layers=[400, 300], a_layers=[400, 300]):
        '''Initialize our agent'''
        # Initialize environment
        self.init_environment(env_id)
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer, self.state_size, self.action_size)
        # Save our variables and initialize our class
        self.tau = tau
        self.gamma = gamma
        self.noise = noise
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        # Keep track of training steps
        self.train_step = 0
        self.env_step = 0

        self.history = {
            'actor_loss': np.empty(0),
            'critic_loss': np.empty(0),
            'reward': np.empty(0)
        }

        # Get our actor and it's target
        self.actor = Actor(self.state_size, self.action_size, max_action=self.max_action, name='actor', layers=a_layers)
        self.actor_target = Actor(self.state_size, self.action_size, max_action=self.max_action, layers=a_layers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lra)
        self.morph(self.actor, self.actor_target, tau=1)

        # Get our first critic and it's agent
        self.critic1 = Critic(self.state_size, self.action_size, name='critic1', layers=c_layers)
        self.critic_target1 = Critic(self.state_size, self.action_size, layers=c_layers)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=lrc)
        self.morph(self.critic1, self.critic_target1, tau=1)

        # Get our second critic and it's target
        self.critic2 = Critic(self.state_size, self.action_size, name='critic2', layers=c_layers)
        self.critic_target2 = Critic(self.state_size, self.action_size, layers=c_layers)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=lrc)
        self.morph(self.critic2, self.critic_target2, tau=1)

        self.fill_replay_buffer(size=warmup)

    def init_environment(self, env_id):
        '''Initialize environment and store it's key properties'''
        # Create and observe environment
        if env_id in ["Pendulum-v1", "Hopper-v2"]:
            self.env = gym.make(env_id)
        else:
            self.env = gym.make(env_id, continuous=True)

        self.monitor_env = Monitor(self.env, "./videos", force=True, video_callable=lambda episode: True)
            
        self.last_state = self.env.reset()
        # Get action space from environment (assume it's low is -high)
        self.max_action = self.env.action_space.high[0]
        self.min_action = self.env.action_space.low[0]
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape[0]

    def step_environment(self, train=True, max_steps=200):
        '''Perform single step in environment'''
        # Choose action to perform
        action = self.choose_action(self.last_state, train=train)
        # Perform the chosen action
        new_state, reward, done, _ = self.env.step(action)
        # Save the action into replay buffer
        self.env_step += 1
        #self.memory.save(self.last_state, new_state, action, reward, done)  TODO Could work?
        self.memory.save(self.last_state, new_state, action, reward, 0 if self.env_step >= done else done)
        # If environment is considered finished, reset it
        self.last_state = new_state
        done = self.env_step >= max_steps or done
        if done:
            self.env_step = 0
            self.last_state = self.env.reset()
        return reward, done

    def morph(self, src, dst, tau=None):
        '''Morph dst weights into src weights (partially)'''
        if not tau:
            tau = self.tau
        for w, tw in zip(src.parameters(), dst.parameters()):
            tw.data.copy_(tau * w.data + (1 - tau) * tw.data)

    def choose_action(self, state, train=True):
        '''Choose action based on observed state (pytorch tensor)'''
        state = torch.FloatTensor(state)
        # Get actions based on actor network and given state and apply noise
        action = self.actor(state).detach().flatten().numpy()
        if train:
            # Apply noise when training, also clip the actions (net could not provide actions of interval, but the noise could)
            action = action + np.random.normal(0, self.noise, size=self.action_size)
            action.clip(action, self.env.action_space.low, self.env.action_space.high)
            self.train_step += 1
        return action

    def evaluate(self, epochs=128, render=False, save=False, fps=24):
        '''Evaluate our agent'''

        env = self.env if not save else self.monitor_env

        # Array to store rewards (probably useless, could just track sum and divide it, whatever)
        rewards = np.empty(0)
        
        # Progress bar
        pbar = tqdm(range(epochs), desc='Evaluating ...', ncols=160)
        for _ in pbar:
            # Initialize per-epoch variables
            reward = 0
            state = env.reset()
            done = False
            while not done:
                if render:          # May not work on your PC :/
                    env.render()
                    sleep(1/fps)
                action = self.choose_action(state, train=False)
                state, r, done, _ = env.step(action)
                reward += r
            rewards = np.append(rewards, reward) # Why? I don't know
            pbar.set_description("Avg: {:.2f}\tLast:{:.2f}".format(rewards.mean(), reward))

    def learn(self, steps, batch_size=100, policy_delay=2):
        '''Train our networks'''
        for step in range(steps):
            self.train_step += 1
            # Obtain samples from replay buffer
            states_, new_states_, actions_, rewards_, dones_ = self.memory.sample(batch_size)
            # Make it tensors
            states, new_states, actions = torch.FloatTensor(states_), torch.FloatTensor(new_states_), torch.FloatTensor(actions_)
            rewards, dones = torch.FloatTensor(rewards_), torch.FloatTensor(1 - dones_).reshape((-1, 1))

            # Get actions from target actor
            target_actions = self.actor_target(new_states) + torch.FloatTensor([np.random.normal(scale=self.noise)])
            target_actions = torch.clip(target_actions, self.min_action, self.max_action)

            # Get q values, choose the minimal (Double Q learning)
            tq1 = self.critic_target1(torch.cat((new_states, target_actions), dim=1))
            tq2 = self.critic_target2(torch.cat((new_states, target_actions), dim=1))

            q1 = self.critic1(torch.cat((states, actions), dim=1))
            q2 = self.critic2(torch.cat((states, actions), dim=1))
            
            tq = torch.min(tq1, tq2)
            tq = rewards.reshape((-1, 1)) + (dones * self.gamma * tq).detach()

            # Calculate loss of both critics
            q1_loss = mse_loss(q1, tq)
            q2_loss = mse_loss(q2, tq)
            self.history['critic_loss'] = np.append(self.history['critic_loss'], (q1_loss+q2_loss).detach())

            # And update their weights
            self.critic_optimizer1.zero_grad()
            self.critic_optimizer2.zero_grad()
            q1_loss.backward()
            q2_loss.backward()
            self.critic_optimizer1.step()
            self.critic_optimizer2.step()

            # Policy gets updated only once per policy_delay steps
            if not bool(step % policy_delay):
                # Calculate actor loss
                new_action = self.actor(states)
                q = self.critic1(torch.cat((states, new_action), dim=1))
                loss = -torch.mean(q)

                for i in range(policy_delay):
                    self.history['actor_loss'] = np.append(self.history['actor_loss'], loss.detach())

                # Calculate actor gradients            
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

                # Update targets
                self.morph_all()

    def morph_all(self):
        '''Apply self.morph method to each network and it's target'''
        self.morph(self.actor, self.actor_target)
        self.morph(self.critic1, self.critic_target1)
        self.morph(self.critic2, self.critic_target2)

    def save(self):
        '''Save network parameters'''
        torch.save(self.actor.state_dict(), os.path.join(self.model_dir, self.actor.name))
        torch.save(self.critic1.state_dict(), os.path.join(self.model_dir, self.critic1.name))
        torch.save(self.critic2.state_dict(), os.path.join(self.model_dir, self.critic2.name))

    def load(self):
        '''Load best performing network parameters'''
        self.actor.load_state_dict(torch.load(os.path.join(self.model_dir, self.actor.name)))
        self.critic1.load_state_dict(torch.load(os.path.join(self.model_dir, self.critic1.name)))
        self.critic2.load_state_dict(torch.load(os.path.join(self.model_dir, self.critic2.name)))

    def fill_replay_buffer(self, size=10000):
        '''At the start, fill the buffer with random steps to explore a little'''
        state = self.env.reset()
        done = False
        for _ in tqdm(range(size), ncols=160, desc='Exploring ...'):
            action = self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            self.memory.save(state, new_state, action, reward, done)
            state = new_state
            if done:
                state = self.env.reset()

    def train(self, steps, batch_size=100, policy_delay=2, r_avg=256, max_steps=1000, pbar_update=5):
        '''The main training loop'''
        done = False
        best = float('-inf')
        env_step_counter = 0
        rewards = np.empty(0)
        reward = 0
        epoch = 0
        self.last_state = self.env.reset()
        pbar = tqdm(range(steps), desc='Initializing training ...', ncols=160)
        for step in pbar:
            if done and env_step_counter:
                # Keep track of records
                rewards = np.append(rewards, reward)
                # Check if best performing agent, if yes - save it
                reward_avg = rewards[-r_avg:].mean()
                self.history['reward'] = np.append(self.history['reward'], reward_avg)

                if reward_avg > best:
                    self.save()
                    best = reward_avg
                # Let's learn the agent
                self.learn(env_step_counter, batch_size=batch_size, policy_delay=policy_delay)
                # Reset control variables                
                reward = 0
                env_step_counter = 0
                epoch += 1
                # Change progress bar
                if not bool(step % pbar_update):
                    pbar.set_description("Avg: {:.2f}   Best:{:.2f}".format(reward_avg, best))
            # If not done, just go through environment
            r, done = self.step_environment(max_steps=max_steps)
            reward += r
            env_step_counter += 1
    
    def get_plottable_data(self, bins, data):
        if len(data) <= bins:
            return data, np.arange(len(data)) + 1
        else:
            drop = len(data) % bins
            values = data[drop:].reshape((bins, -1)).sum(axis=1) / bins
            ticks = (np.arange(len(values)) + 1) * (len(data)) // bins
            return values, ticks
    
    def plot(self, save, show, size, bins=128):
        '''Plot training info'''
        a_vals, a_ticks = self.get_plottable_data(bins, self.history['actor_loss'])
        c_vals, c_ticks = self.get_plottable_data(bins, self.history['critic_loss'])
        r_vals, r_ticks = self.get_plottable_data(bins, self.history['reward'])

        sns.set_theme()

        fig, (ax_a, ax_c, ax_r) = plt.subplots(1, 3, figsize=size)

        sns.lineplot(x=a_ticks, y=a_vals, ax = ax_a, color="red")
        ax_a.set_title("Actor Loss")
        ax_a.set_xlabel("Steps")

        sns.lineplot(x=c_ticks, y=c_vals, ax = ax_c, color="green")
        ax_c.set_title("Critic Loss")
        ax_c.set_xlabel("Steps")

        sns.lineplot(data=self.history["reward"], ax = ax_r, color="blue")
        ax_r.set_title("Average Reward")
        ax_r.set_xlabel("Episodes")
        
        plt.tight_layout(pad=0.5)
        if save:
            fig.savefig(save)
        if show:
            plt.show()