import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


# 1. LSTM-Actor模型
class LSTMActor(nn.Module):
    def __init__(self, state_dim=500, action_dim=10, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_dim)
        self.hidden_size = hidden_size

    def forward(self, state, hidden=None):
        # 输入形状处理
        if hidden is None:
            h = torch.zeros(1, 1, self.hidden_size)
            c = torch.zeros(1, 1, self.hidden_size)
            hidden = (h, c)

        # LSTM时序处理
        state = state.unsqueeze(1)  # (batch, seq=1, state_dim)
        lstm_out, new_hidden = self.lstm(state, hidden)

        # 输出动作分布
        logits = self.fc(lstm_out[:, -1, :])
        return torch.distributions.Categorical(logits=logits), new_hidden


# 2. LSTM-Critic模型
class LSTMCritic(nn.Module):
    def __init__(self, state_dim=500, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, state, hidden=None):
        if hidden is None:
            h = torch.zeros(1, 1, self.hidden_size)
            c = torch.zeros(1, 1, self.hidden_size)
            hidden = (h, c)

        state = state.unsqueeze(1)
        lstm_out, new_hidden = self.lstm(state, hidden)
        value = self.fc(lstm_out[:, -1, :])
        return value.squeeze(-1), new_hidden


# 3. 升级版PPO智能体（含LSTM状态管理）
class LSTMPPOAgent:
    def __init__(self, state_dim=500, action_dim=10):
        # 超参数保持不变
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_epochs = 4
        self.clip_epsilon = 0.2
        self.batch_size = 64
        self.buffer_size = 2048

        # 模型与优化器
        self.actor = LSTMActor(state_dim, action_dim)
        self.critic = LSTMCritic(state_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 经验池结构扩展：存储LSTM隐藏状态
        self.buffer = deque(
            maxlen=self.buffer_size)  # 存储元组：(state, action, reward, next_state, done, log_prob, value, h_actor, c_actor, h_critic, c_critic)

    # 4. 数据收集（需记录LSTM状态）
    def collect_experience(self, transition):
        self.buffer.append(transition)

    # 5. 带LSTM状态的交互逻辑
    def interact_with_env(self, env, max_steps=200):
        state = env.reset()
        episode_reward = 0

        # 初始化LSTM隐藏状态
        h_actor, c_actor = torch.zeros(1, 1, self.actor.hidden_size), torch.zeros(1, 1, self.actor.hidden_size)
        h_critic, c_critic = torch.zeros(1, 1, self.critic.hidden_size), torch.zeros(1, 1, self.critic.hidden_size)

        for _ in range(max_steps):
            # 转换为Tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Actor预测
            with torch.no_grad():
                dist, (new_h_actor, new_c_actor) = self.actor(state_tensor, (h_actor, c_actor))
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # Critic预测
                value, (new_h_critic, new_c_critic) = self.critic(state_tensor, (h_critic, c_critic))

            # 执行动作
            next_state, reward, done, _ = env.step(action.item())

            # 存储带LSTM状态的数据
            transition = (
                state.copy(),
                action.item(),
                reward,
                next_state.copy(),
                done,
                log_prob.item(),
                value.item(),
                h_actor.numpy().copy(),  # 保存当前步的隐藏状态
                c_actor.numpy().copy(),
                h_critic.numpy().copy(),
                c_critic.numpy().copy()
            )
            self.collect_experience(transition)

            # 更新隐藏状态
            h_actor, c_actor = new_h_actor, new_c_actor
            h_critic, c_critic = new_h_critic, new_c_critic

            episode_reward += reward
            state = next_state

            if done or len(self.buffer) >= self.buffer_size:
                break

        return episode_reward

    # 6. 训练数据准备（含LSTM状态恢复）
    def prepare_training_data(self):
        # 解压缓冲区数据
        states, actions, rewards, next_states, dones, old_log_probs, old_values, h_actors, c_actors, h_critics, c_critics = zip(
            *self.buffer)

        # 转换为Tensor
        states = torch.FloatTensor(np.array(states))  # (buffer_size, 500)
        actions = torch.LongTensor(np.array(actions))  # (buffer_size,)
        rewards = torch.FloatTensor(np.array(rewards))  # (buffer_size,)
        next_states = torch.FloatTensor(np.array(next_states))  # (buffer_size, 500)
        dones = torch.FloatTensor(np.array(dones))  # (buffer_size,)
        old_log_probs = torch.FloatTensor(old_log_probs)  # (buffer_size,)
        old_values = torch.FloatTensor(old_values)  # (buffer_size,)

        # 处理LSTM隐藏状态
        h_actors = torch.FloatTensor(np.array(h_actors))  # (buffer_size, 1, hidden)
        c_actors = torch.FloatTensor(np.array(c_actors))
        h_critics = torch.FloatTensor(np.array(h_critics))
        c_critics = torch.FloatTensor(np.array(c_critics))

        # 计算GAE和Returns（需考虑LSTM时序依赖）
        with torch.no_grad():
            values, _ = self.critic(states, (h_critics, c_critics))
        next_values, _ = self.critic(next_states, (h_critics, c_critics))  # 近似处理

        # GAE计算（需要调整处理方式）
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        advantages = torch.tensor(advantages)

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, old_log_probs, returns, advantages, h_actors, c_actors, h_critics, c_critics

    # 7. 分步训练（考虑LSTM序列）
    def update_models(self):
        states, actions, old_log_probs, returns, advantages, h_actors, c_actors, h_critics, c_critics = self.prepare_training_data()

        # 创建序列索引（保持时序关系）
        indices = np.arange(len(states))

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                # 提取批次数据
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # --- Actor训练（带LSTM状态）---
                # 重组LSTM隐藏状态
                batch_h_actors = h_actors[batch_idx].transpose(0, 1).contiguous()
                batch_c_actors = c_actors[batch_idx].transpose(0, 1).contiguous()

                dist, _ = self.actor(batch_states, (batch_h_actors, batch_c_actors))
                new_log_probs = dist.log_prob(batch_actions)

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 熵正则化
                entropy = dist.entropy().mean()

                self.actor_optim.zero_grad()
                (actor_loss - 0.01 * entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

                # --- Critic训练（带LSTM状态）---
                batch_h_critics = h_critics[batch_idx].transpose(0, 1).contiguous()
                batch_c_critics = c_critics[batch_idx].transpose(0, 1).contiguous()

                current_values, _ = self.critic(batch_states, (batch_h_critics, batch_c_critics))
                critic_loss = 0.5 * (current_values - batch_returns).pow(2).mean()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

        self.buffer.clear()


# 8. 使用示例
if __name__ == "__main__":
    class MicroserviceEnv:  # 需替换为实际环境
        def reset(self):
            return np.random.randn(500)

        def step(self, action):
            return np.random.randn(500), np.random.randn(), False, {}


    env = MicroserviceEnv()
    agent = LSTMPPOAgent()

    for episode in range(1000):
        episode_reward = agent.interact_with_env(env)
        if len(agent.buffer) >= agent.buffer_size:
            agent.update_models()
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")