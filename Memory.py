import random
import numpy as np
from utils.SumTree import SumTree
# Deep Learning Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import normalize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from collections import deque

class MemoryReply:
    '''Defines the memory replay of stored samples.
    memory_size (int) : maximum size of buffer
    replay_size (int) : size of each training batch
    seed        (int) : random seed
    '''
    def __init__(self, memory_size, replay_size, seed=958):
        self.memory = deque(maxlen=memory_size)  
        self.replay_size = replay_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        '''Store samples to memory
        state      ([float]) : The current state space of the givern envirnment
        action         (int) : The stochastic or predicted action for the current state space
        reward         (int) : The reward recieved for that action
        next_state ([float]) : The next state space of the givern envirnment after an action has been taken
        done          (bool) : Whether the envirnment has been completed or not
        '''
        for s,a,r,ns,d in zip(state, action, reward, next_state, done):
            self.memory.append({"state":s, "action":a, "reward":r, "next_state":ns, "done":d})
    
    def sample(self):
        '''Sample experiences from memory.'''
        experiences = random.sample(self.memory, k=self.replay_size)

        states = torch.FloatTensor(np.vstack([e['state'] for e in experiences])).to(device)
        actions = torch.FloatTensor(np.array([e['action'] for e in experiences])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([e['reward'] for e in experiences])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.vstack([e['state'] for e in experiences])).to(device)
        dones = torch.FloatTensor(np.array([float(e['done']) for e in experiences])).unsqueeze(1).to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
def append_sample(state, actions, rewards, next_state, dones, local_net , target_net,discount_factor,memory):
    target = local_net(state).data
    old_val = target.gather(1,actions.long().unsqueeze(1))
    target_val = target_net(next_state).data

    r = rewards + discount_factor * target_val.argmax(dim=1)

    for en,idx in enumerate(actions):
        target[en][actions.long()[en]] = r[en]

    error = abs(old_val - target.gather(1,actions.long().unsqueeze(1)))
    
    memory.add(error, (state, actions, rewards, next_state, dones))
    

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        if is_weight.max() == 0:
            is_weight /= np.finfo(np.float32).eps
        else:
            is_weight /= (is_weight.max()+np.finfo(np.float32).eps)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)