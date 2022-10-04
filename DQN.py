import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import normalize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

def train_model(local_net,target_net,memory,replay_size,action_size,discount_factor,optimizer,tau=0.001):

    states, actions, rewards, next_states, dones = memory.sample()

    # Q function of next state
    next_states = torch.Tensor(next_states)

    Q_targets = target_net(next_states).data

    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    Q_targets = (rewards + (1 - dones) * discount_factor).flatten(0)* Q_targets.max(1)[0]

    Q_expected = local_net(states).gather(1, actions.long())

    optimizer.zero_grad()

    # MSE Loss function
    loss =  F.mse_loss(Q_expected, Q_targets.unsqueeze(1)).mean()
    loss.backward()
    
    # clip gradient
    #torch.nn.utils.clip_grad_norm(local_net.parameters(), 1)
    
    # and train
    optimizer.step()
    
    #soft Update
    local_net,target_net = soft_update(local_net,target_net,tau)
    
    return local_net

def learn(experiences,local_net,target_net,tau= 0.001,gamma=0.95):
    '''Update value parameters using given batch of experience tuples.
    experiences (tuple) : state, action, reward, next_state, done

    Soft update: θ_target = τ * θ_local + (1 - τ) * θ_target
    '''
    states, actions, rewards, next_states, dones = experiences

    # Get max predicted Q values (for next states) from target model
    Q_targets = target_net(next_states).detach().max(1)[0].unsqueeze(1)

    # Compute Q targets for current states 
    Q_targets = rewards + (gamma * Q_targets * (1 - dones))

    # Get expected Q values from local model
    Q_expected = local_net(states).gather(1, actions.type(torch.int64))

    # Compute loss
    loss = F.mse_loss(Q_expected, Q_targets)

    # Minimize the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    local_net,target_net = soft_update(local_net,target_net,tau)
    
    return local_net,target_net
    
def soft_update(local_net,target_net,tau):
    # Soft update target network
    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    return local_net,target_net