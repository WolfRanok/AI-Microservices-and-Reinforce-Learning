import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from ENV_and__important_calculate import *
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lstm_hidden=128, fc_hidden=64):
        super(Actor, self).__init__()
        # LSTM层处理时序状态
        self.lstm = nn.LSTM(state_dim, lstm_hidden, batch_first=True)
        # 全连接层生成动作概率
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, action_dim),
        )

    def forward(self, state, hidden=None):
        # 输入形状处理: (batch_size, state_dim) -> (batch_size, 1, state_dim)
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        # LSTM处理时序特征
        lstm_out, hidden = self.lstm(state, hidden)
        # 取最后一个时间步输出
        lstm_out = lstm_out[:, -1, :]
        # 生成动作概率分布
        action_probs = self.fc(lstm_out)
        # action_probs = action_probs.view(-1,MA_AIMS_NUM,NODE_NUM)
        action_probs = F.softmax(action_probs, dim=-1)
        return action_probs, hidden
# 2. 定义Critic网络（Q值评估）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lstm_hidden=128, fc_hidden=64):
        super(Critic, self).__init__()
        # LSTM处理状态
        self.lstm = nn.LSTM(state_dim, lstm_hidden, batch_first=True)
        # 融合状态和动作特征
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + action_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1)
        )

    def forward(self, state, action, hidden=None):
        # 输入形状处理
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        # LSTM处理状态
        lstm_out, hidden = self.lstm(state, hidden)
        lstm_out = lstm_out[:, -1, :]
        # 拼接状态特征和动作
        action = action.reshape(len(action),-1)
        combined = torch.cat([lstm_out, action], dim=1)
        # 评估Q值
        q_value = self.fc(combined)
        return q_value, hidden
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, lstm_hidden=128, fc_hidden=128):
#         super(Actor, self).__init__()
#         # 全连接层生成动作概率
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, fc_hidden),
#             nn.ReLU(),
#             nn.Linear(fc_hidden, action_dim),
#             nn.Tanh()
#         )
#
#     def forward(self, state, hidden=None):
#         # 输入形状处理: (batch_size, state_dim) -> (batch_size, 1, state_dim)
#         if len(state.shape) == 2:
#             state = state.unsqueeze(1)
#         output = self.fc(state)
#         # 生成每个服务的部署概率分布
#         # decisions = [head(features) for head in self.decision_heads]
#         output = F.softmax(output, dim=-1)
#         return output  # shape: (batch, service_num, server_num)
# # 2. 定义Critic网络（Q值评估）
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, lstm_hidden=128, fc_hidden=128):
#         super(Critic, self).__init__()
#         # 融合状态和动作特征
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim + action_dim, fc_hidden),
#             nn.ReLU(),
#             nn.Linear(fc_hidden, 1)
#         )
#
#     def forward(self, state, action, hidden=None):
#         # 输入形状处理
#         # if len(state.shape) == 2:
#         #     state = state.unsqueeze(1)
#         if len(action.shape) == 3:
#             action = action.squeeze(1)
#         # 拼接状态特征和动作
#         # action = action.reshape(len(action),-1)
#         combined = torch.cat([state, action], dim=-1)
#         # 评估Q值
#         q_value = self.fc(combined)
#         return q_value
# class OUNoise:
#         def __init__(self, size, mu=0.0, theta=0.15, sigma=0):
#             self.mu = mu * np.ones(size)
#             self.theta = theta
#             self.sigma = sigma
#             self.state = self.mu.copy()
#
#         def reset(self):
#             self.state = self.mu.copy()
#
#         def sample(self):
#             dx = self.theta * (self.mu - self.state)
#             dx += self.sigma * np.random.randn(len(self.state))
#             self.state += dx
#             return self.state

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # if len(action.shape) == 2:
        #     action = action.squeeze(0)
        self.buffer.append((
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.FloatTensor([done])
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )
    def clean_buffer(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class LSTM_DDPG_Agent:
    def __init__(self, ms_aims_num, node_num, user_num):
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.01  # 目标网络软更新系数
        self.batch_size = 64  # 训练批次大小
        self.buffer_size = 1000  # 经验池容量
        self.noise_scale = 0.1  # 噪声强度
        # self.state_dim = ms_aims_num * node_num \
        #              + 3 * 2 * node_num \
        #              + user_num * (node_num + 1) * node_num * ms_aims_num \
        #              + user_num * ms_aims_num \
        #              + 3 * ms_aims_num \
        #              + ms_aims_num * ms_aims_num \
        #              + node_num * node_num
        # self.state_dim = ms_aims_num * node_num \
        #                  + 3 * 2 * node_num \
        #                  + user_num * ms_aims_num \
        #                  + 3 * ms_aims_num \
        #                  + ms_aims_num * ms_aims_num \
        #                  + node_num * node_num
        self.state_dim = ms_aims_num * node_num \
                         + 3 * 2 * node_num \
                         + 8\
                         + node_num*4\
                         + ms_aims_num * ms_aims_num \
                         + node_num * node_num
        self.action_dim = node_num

        # 主网络
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)

        # 目标网络（延迟更新）
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 经验回放池
        self.buffer = ReplayBuffer(100000)


    # 5. 动作选择（添加噪声）
    def select_action(self, state, hidden=None, explore=True):
        with torch.no_grad():
            # 生成动作概率分布
            probs, new_hidden = self.actor(state.unsqueeze(0), hidden)
            # 训练时加入Gumbel噪声探索
            if explore:
                action = torch.nn.functional.gumbel_softmax(probs, hard=True)
            # 测试时直接取最大概率
            else:
                action = torch.argmax(probs, dim=-1)
                action = torch.nn.functional.one_hot(action, num_classes=probs.shape[-1]).float()
        return action.squeeze(0), new_hidden

    # 7. 训练步骤
    def update_model(self, batch_size,data):
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # Critic网络更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Actor网络更新
        actor_actions = self.actor(states)
        actor_loss = self.critic(states, actor_actions)
        actor_loss = -actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # 目标网络软更新
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            # 记录误差总和，与参数改变量总和
        data['loss'] += critic_loss.item()
        data['param_change'] += sum((abs(param.grad.norm().item()) for param in self.actor.parameters()))

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)



if __name__ == '__main__':
    # 输入数据示例
    # env = MicroserviceEnv()
    state = initial_state()
    state = torch.tensor(state,dtype=torch.float32)
    # state = state.unsqueeze(0)
    agent = LSTM_DDPG_Agent(MA_AIMS_NUM,NODE_NUM,USER_NUM)
    actor = agent.actor
    critic = agent.critic

    # 前向传播
    action_probabilities = actor(state)
    print(action_probabilities)
    action_value = critic(state, action_probabilities)
    # print(f"隐藏层层数分别为:actor {ha},critic {hc}")

    print("actor网络的行动输入如下所示(Actor Output):")
    print(action_probabilities)
    print("\ncritic网络对该行动给出的评价 (Critic Output):")
    print(action_value)
    # action,_ = agent.select_action(state)
    # print(action)
    # print(agent.select_action(state),_)

