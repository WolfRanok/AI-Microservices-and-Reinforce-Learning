import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym

# 超参数
GAMMA = 0.99  # 折扣因子
TAU = 0.005  # 目标网络软更新系数
LR_ACTOR = 1e-4  # Actor 学习率
LR_CRITIC = 1e-3  # Critic 学习率
BUFFER_SIZE = 100000  # 经验池大小
BATCH_SIZE = 64  # 批量大小
EPSILON = 0.1  # ε-贪婪策略的随机性

# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state):
        return self.net(state)  # 输出每个离散动作的评分

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# 经验回放池
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)

# DDPG Agent for Discrete Actions
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 初始化 Actor 和 Critic 网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, epsilon=EPSILON):
        """ε-贪婪策略选择动作"""
        if np.random.random() < epsilon:
            return np.random.choice(self.action_dim)  # 随机选择动作
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # 批量维度
            q_values = self.actor(state).detach().cpu().numpy()[0]
            return np.argmax(q_values)  # 选择 Q 值最大的动作

    def train(self):
        """训练 Actor 和 Critic"""
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Critic 训练
        one_hot_actions = torch.zeros(BATCH_SIZE, self.action_dim)
        one_hot_actions[range(BATCH_SIZE), actions] = 1.0

        target_q_values = rewards + (1 - dones) * GAMMA * self.critic_target(next_states, one_hot_actions).detach()
        current_q_values = self.critic(states, one_hot_actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 训练
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# 训练主函数
def train_ddpg_discrete(env_name, episodes=500, max_steps=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # 动作空间维度

    agent = DDPGAgent(state_dim, action_dim)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 根据 ε-贪婪策略选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add((state, action, reward, next_state, float(done)))

            state = next_state
            episode_reward += reward

            # 训练智能体
            agent.train()

            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()

# 运行训练
train_ddpg_discrete('CartPole-v1', episodes=200)
