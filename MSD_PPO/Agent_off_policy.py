import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


import network
from replay_buffer import ReplayBuffer
torch.autograd.set_detect_anomaly(True)  # 启用异常检测[5,6](@ref)

class PPOAgent:
    def __init__(self, state_dim, action_dim, action_type='binary',  # 修正参数名
                 gamma=0.99, lambad_=0.5, clip_range=0.2, ppo_epochs=10,
                 tau=0.005, value_coeff=0.25):
        # 根据action_type选择网络结构
        if action_type == 'continuous':
            raise NotImplementedError("Continuous action not supported in current setup")

        self.actor = network.LSTM_Actor(state_dim, action_dim)
        self.actor_target = network.LSTM_Actor(state_dim, action_dim)
        self.critic = network.LSTM_Critic(state_dim)
        self.critic_target = network.LSTM_Critic(state_dim)

        self.ACNet = network.MLP_ACNet(state_dim,action_dim)

        self.gamma = gamma
        self.lambad_ = lambad_
        self.clip_range = clip_range
        self.ppo_epochs = ppo_epochs
        self.tau = tau
        self.value_coeff = value_coeff
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.ACNet_optimizer = optim.Adam(self.ACNet.parameters(), lr= 3e-4)


        # Initialize target networks
        self._update_target_networks()
        self.buffer = ReplayBuffer(max_capacity=1000)

    def get_action(self,state):
        action = self.actor.select_action(state)
        return action
    def ACNet_get_action(self,state):
        action = self.ACNet.select_action(state)
        return action
    def _update_target_networks(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.append(R)
        return torch.tensor(returns[::-1], dtype=torch.float32)

    def compute_gae(self, states, rewards, dones):
        advantages = []
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                advantage = (1 - self.lambad_) * rewards[t]
            else:
                next_value = self.critic_target(states[t+1]).to(self.device)
                advantage= (1 - self.lambad_) * (rewards[t] + self.gamma * (1 - dones[t+1]) * next_value)\
                            + self.lambad_ * self.gamma * (1 - dones[t+1]) * advantages[-1]
            advantages.append(advantage)
        return torch.tensor(advantages[::-1], dtype=torch.float32)

    def compute_gae_new(self, states, rewards, dones):
        advantages = np.zeros(len(states),dtype=np.float32)
        for t in range(len(states)):
            discount = 1
            if t == len(states)-1:
                advantages[t] = rewards[t]-self.critic(states[t])
            for k in range(t,len(states)-1):
                value = self.critic(torch.tensor(states[k], dtype=torch.float32))
                next_value = self.critic(torch.tensor(states[k+1], dtype=torch.float32))
                delat = rewards[k] + self.gamma*(1-dones[k])*next_value - value
                advantages[t] += discount * delat
                discount = self.gamma*self.lambad_
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def update_model(self, batch_size):
        states, actions, rewards, next_states, dones, returns, advantages = self.buffer.sample(batch_size)

        actions = actions.squeeze(1)
        returns = returns.squeeze(1)
        advantages = advantages.squeeze(1)
        # # 标准化
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # 计算当前值函数和优势函数（使用GAE）

        # returns = self.compute_returns(rewards, dones)
        # advantages = self.compute_gae(rewards, states, dones)
        # 计算旧策略的对数概率
        with torch.no_grad():
            actions = actions.squeeze().long().unsqueeze(1)
            assert actions.shape == (len(states), 1), f"Actions shape error: {actions.shape}"
            # prob = self.actor(states)
            prob,_ = self.ACNet(states)
            old_log_probs = torch.log(prob.gather(1, actions))
        # # PPO多epoch优化
        pollicy_losses = np.zeros(self.ppo_epochs)
        value_losses = np.zeros(self.ppo_epochs)
        total_losses = np.zeros(self.ppo_epochs)
        for p in range(self.ppo_epochs):
            actions = actions.squeeze().long().unsqueeze(1)
            assert actions.shape == (len(states), 1), f"Actions shape error: {actions.shape}"
            prob,_ = self.ACNet(states)
            new_log_probs = torch.log(prob.gather(1, actions))
            # # 在训练循环中添加
            # print("Old Log Probs Mean:", old_log_probs.mean())
            # print("New Log Probs Mean:", new_log_probs.mean())
            # print("Advantages Mean:", advantages.mean())
            # 计算比率
            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped_ratios = ratios.clamp(1 - self.clip_range, 1 + self.clip_range)
            # 构建损失函数
            # policy_loss = -(ratios * advantages + (1 - self.clip_range) * clipped_ratios * advantages).mean()
            current_values = self.critic(states)
            policy_loss = -torch.mean(torch.min(ratios * advantages, clipped_ratios * advantages))
            value_loss = nn.MSELoss()(current_values, returns)
            total_loss = policy_loss + self.value_coeff * value_loss
            pollicy_losses[p] = policy_loss
            value_losses[p] = value_loss
            total_losses[p] = total_loss
            # 反向传播更新网络
            # self.actor_optimizer.zero_grad()
            # self.critic_optimizer.zero_grad()
            # policy_loss.backward(retain_graph=True)
            # value_loss.backward(retain_graph=True)
            # self.actor_optimizer.step()
            # self.critic_optimizer.step()
            self.ACNet_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.ACNet_optimizer.step()
            # with torch.no_grad():
            #     # 打印Actor/Critic参数的梯度范数
            #     actor_grad_norm = 0.0
            #     for param in self.actor.parameters():
            #         if param.grad is not None:
            #             actor_grad_norm += param.grad.norm().item()
            #     critic_grad_norm = 0.0
            #     for param in self.critic.parameters():
            #         if param.grad is not None:
            #             critic_grad_norm += param.grad.norm().item()
            #     print(f"Actor Grad Norm: {actor_grad_norm:.4f}, Critic Grad Norm: {critic_grad_norm:.4f}")

        # 更新目标网络
        # self._update_target_networks()
        return pollicy_losses.mean(), value_losses.mean(), total_losses.mean()

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            for batch in data_loader:
                states, actions, rewards, next_states, dones = batch
                self.update_model(len(states))
            print(f"Epoch {epoch + 1}/{num_epochs} completed")