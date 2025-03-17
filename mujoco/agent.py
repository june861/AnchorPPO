import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, policy_layers, sample_action_num = 1):
        super().__init__()
        self.state_dim = np.array(envs.single_observation_space.shape).prod()
        self.action_dim = np.array(envs.single_action_space.shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # If 3 layers
        if policy_layers == 3:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(self.state_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, self.action_dim), std=0.01)
            )

        # If 7 layers
        if policy_layers == 7:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(self.state_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, self.action_dim), std=0.01)
            )
        self.actor_log_std = nn.Parameter(torch.zeros(1, self.action_dim))
        
        self.sample_action_num = sample_action_num

    def get_logprobs(self, actions, probs):
        """ actions shape is [num_envs, self.sample_action_num, action_dim] """
        log_probs = []
        for i in range(self.sample_action_num):
            log_probs.append(probs.log_prob(actions[:,i,:]))
        
        return torch.stack(log_probs, dim = 1).sum(2)


    def get_value(self, s):
        return self.critic(s)

    # junweiluo: 增加函数
    def sample_action(self, probs):
        actions = []
        for _ in range(self.sample_action_num):
            i_action = probs.sample()
            actions.append(i_action)
        actions = torch.stack(actions, dim = 1)
        log_probs =  self.get_logprobs(actions, probs)
        return actions, log_probs

    def get_action_and_value(self, s, a=None):
        action_mean = self.actor_mean(s)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if a is None:
            a, log_probs = self.sample_action(probs)
            return a, log_probs, probs.entropy().sum(-1), self.critic(s), probs

        log_probs = self.get_logprobs(a, probs)
        return a, log_probs, probs.entropy().sum(-1), self.critic(s), probs
