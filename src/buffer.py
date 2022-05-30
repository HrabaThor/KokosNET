"""
    File: buffer.py 
    Project: KNN - Actor-Critic 
    Authors: Martin Kostelník (xkoste12), Michal Glos (xglosm01), Michal Szymik (xszymi00) 
    Date: 2022-05-30

    Description: This file contains definition of ReplayBuffer class that is
    used for agent training. Using this interface one can save and sample data
    from the buffer.

"""
import numpy as np

'''
This module provides a very simple implementation of replay buffer
'''

class ReplayBuffer:
    '''
    Simple buffer storing the historical data of agent performing in the environment

    Provides:
        Saving data after each step
        Sampling historical data
    '''
    def __init__(self, size, state, actions):
        '''Initialize the buffer'''
        # Those should be tuples in order to unpack them later
        if not isinstance(actions, tuple):
            actions = (actions,)
        if not isinstance(state, tuple):
            state = (state,)

        # Initilize and store configuration parameters
        self.pointer = 0
        self.size = size
        self.state = state

        # Initialize the memory
        self.states = np.zeros((self.size, *self.state), dtype=float)
        self.new_states = np.zeros((self.size, *self.state), dtype=float)
        self.rewards = np.zeros((self.size))
        self.actions = np.zeros((self.size, *actions), dtype=float)
        self.dones = np.zeros((self.size), dtype=bool)

    
    def save(self, state, new_state, action, reward, done):
        '''Save data on top of our buffer'''
        # Update pointer
        self.pointer += 1
        # Point to the first empty space of buffers
        pointer = self.pointer % self.size
        # Save data
        self.states[pointer] = state
        self.new_states[pointer] = new_state
        self.rewards[pointer] = reward
        self.actions[pointer] = action
        self.dones[pointer] = done


    def sample(self, size):
        '''Retrieve a stochastic minibacth of our data'''
        # Ensure the required batch is not bigger then possible
        max_samples = min(self.pointer, self.size)
        # Randomly choose our minibacth
        minibatch = np.random.choice(max_samples, size)
        # Return the sample in order as save method
        return self.states[minibatch], self.new_states[minibatch], \
            self.actions[minibatch], self.rewards[minibatch], self.dones[minibatch]
