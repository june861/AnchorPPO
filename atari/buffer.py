import numpy as np
import torch


class Buffer:
    def __init__(self, num_steps, num_envs, observation_shape, device, discrete_action_n):
        self.states = np.zeros((num_steps, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.flags = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.step = 0
        self.num_steps = num_steps
        self.device = device
        # junweiluo：增加一个收录全部动作的buffer
        self.total_logits = np.zeros((num_steps, num_envs, discrete_action_n))
        self.discrete_action_shape = discrete_action_n

    def push(self, state, action, reward, flag, log_prob, value, total_logits):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.flags[self.step] = flag
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        # junweiluo
        self.total_logits[self.step] = total_logits
        
        self.step = (self.step + 1) % self.num_steps

    def get(self):
        return (
            torch.from_numpy(self.states).to(self.device),
            torch.from_numpy(self.actions).to(self.device),
            torch.from_numpy(self.rewards).to(self.device),
            torch.from_numpy(self.flags).to(self.device),
            torch.from_numpy(self.log_probs).to(self.device),
            torch.from_numpy(self.values).to(self.device),
            # junweiluo
            torch.from_numpy(self.total_logits).to(self.device),
        )
