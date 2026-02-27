import csv
import os
import numpy as np
import torch
import torch.nn.functional as F

from Environment import ENV
import Arguments
from network import DQN_QNetwork, ReplayBuffer


# 替代TensorBoard的日志记录类
class DummyWriter:
    def add_scalar(self, name, value, step):
        print(f"episode {step}: {name} = {value:.4f}")


# 模型保存/加载
def save_model(agent, model_path):
    os.makedirs(model_path, exist_ok=True)
    torch.save(agent.target_net.state_dict(), os.path.join(model_path, "dqn_target_net_length_of_service_2_4.pth"))
    print("模型已保存！")


# 日志写入函数
def write_training_data(path, episode, q_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["episode", "q_loss"])
        writer.writerow([episode, q_loss])


def write_evaluate_data(path, evaluate_num, evaluate_reward, evaluate_sum_delay):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["evaluate_num", "evaluate_reward", "evaluate_sum_delay"])
        writer.writerow([evaluate_num, evaluate_reward, evaluate_sum_delay])


# DQN智能体类
class DQNAgent:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.action_dim = env.NODE_NUM
        self.device = env.device

        # 初始化Q网络和目标网络
        self.q_net = DQN_QNetwork(env, args, self.action_dim).to(self.device)
        self.target_net = DQN_QNetwork(env, args, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())  # 初始同步参数

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(max_size=env.Ms_instance_sum*5)

        # 优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)

        # ε-贪心参数
        self.epsilon = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay

    def get_action(self, state, action_mask, train=True):
        # ε-贪心策略选择动作
        if train and np.random.rand() < self.epsilon:
            # 随机动作（考虑动作掩码）
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        else:
            # 基于Q值的贪心动作
            with torch.no_grad():
                # 确保state格式正确（与PPO一致）
                mlp_state, deployment, rout, ms_dependency = state[0]
                q_values = self.q_net([(mlp_state, deployment, rout, ms_dependency)], self.env)
                q_values = q_values.masked_fill(torch.tensor(action_mask, device=self.device) == 0, -np.inf)
                return torch.argmax(q_values, dim=1).item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)

    def update_target_net(self):
        # 硬更新目标网络
        if self.env.global_step % self.args.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def learn(self):
        if len(self.replay_buffer.buffer) < self.args.batch_size:
            return 0.0

        # 采样批次数据
        samples = self.replay_buffer.sample(self.args.batch_size)
        states, actions, rewards, next_states, dones, action_masks = zip(*samples)

        # 转换为张量
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)

        # 计算当前Q值和目标Q值
        current_q = self.q_net(states, self.env).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states, self.env)
            # 处理动作掩码
            action_mask_tensor = torch.tensor(action_masks, device=self.device)
            next_q = next_q.masked_fill(action_mask_tensor == 0, -np.inf)
            max_next_q = torch.max(next_q, dim=1, keepdim=True)[0]
            target_q = rewards + self.args.gamma * max_next_q * (1 - dones)

        # 计算损失并优化
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# 训练主流程
def train(env, args):
    # 创建日志目录
    os.makedirs(os.path.dirname('Environmental_parameters/length_of_service/Training_data/dqn_training_length_of_service_2_4.csv'), exist_ok=True)
    os.makedirs(os.path.dirname('Environmental_parameters/length_of_service/Evaluation_data/dqn_evaluate_length_of_service_2_4.csv'), exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    agent = DQNAgent(env, args)
    writer = DummyWriter()
    total_episodes = 0
    best_reward = -np.inf
    best_delay = np.inf  # 新增：记录最佳时延

    while total_episodes < args.max_train_episodes:
        state = env.reset()
        episode_reward = 0
        done = False
        episode_losses = []  # 存储本轮所有损失值

        # 一轮采样
        while not done:
            # 提取动作掩码
            action_mask = state[0][10:]
            action = agent.get_action([state], action_mask)

            next_state, reward, done = env.agent_step(action)
            episode_reward += reward

            # 存储经验到缓冲区
            agent.replay_buffer.store(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                action_mask=action_mask
            )

            state = next_state
            env.global_step += 1

            # 定期更新网络
            if env.global_step % args.update_freq == 0:
                loss = agent.learn()
                agent.update_target_net()
                episode_losses.append(loss)  # 存储本轮的损失值，不立即记录到日志

        total_episodes += 1
        # 计算本轮平均损失
        if episode_losses:
            avg_episode_loss = sum(episode_losses) / len(episode_losses)
            print('=' * 20)
            writer.add_scalar("Episode Loss/Q_Loss", avg_episode_loss, total_episodes)
            write_training_data('Environmental_parameters/length_of_service/Training_data/dqn_training_length_of_service_2_4.csv', total_episodes, avg_episode_loss)

        # 更新ε和记录日志
        agent.update_epsilon()
        writer.add_scalar("Episode Reward", episode_reward, total_episodes)

        # 每轮训练后立即测试时延
        test_reward, test_sum_delay, _, _ = evaluate_policy(env, agent, args)
        writer.add_scalar("Test Sum Delay", test_sum_delay, total_episodes)
        writer.add_scalar("Test Reward", test_reward, total_episodes)

        # # 保存最佳时延模型
        # if test_sum_delay < best_delay:
        #     save_model(agent, args.model_path)
        #     best_delay = test_sum_delay
        #     print(f"新的最佳时延模型已保存: {test_sum_delay:.2f}")

        # 定期评估与保存模型（保持原有逻辑）
        if total_episodes % args.evaluate_freq == 0:
            evaluate_num = total_episodes // args.evaluate_freq
            evaluate_reward, evaluate_sum_delay, _, _ = evaluate_policy(env, agent, args, evaluate_num)
            writer.add_scalar("Evaluate Reward", evaluate_reward, total_episodes)
            writer.add_scalar("Evaluate Sum Delay", evaluate_sum_delay, total_episodes)

            if evaluate_reward > best_reward:
                save_model(agent, args.model_path)
                best_reward = evaluate_reward

            print(f"Episode {total_episodes}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")


# 评估策略（记录奖励和时延）
def evaluate_policy(env, agent, args, evaluate_num=None):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    rewards = []

    while not done:
        steps += 1
        action_mask = state[0][10:]
        action = agent.get_action([state], action_mask, train=False)  # 关闭探索
        next_state, reward, done = env.agent_step(action)
        rewards.append(reward)
        total_reward += reward
        state = next_state

    # 记录评估指标
    evaluate_reward = total_reward
    evaluate_sum_delay = env.last_episode_sum_delay
    avg_reward = total_reward / steps
    avg_delay = evaluate_sum_delay / steps

    # 写入评估数据
    if evaluate_num is not None:
        write_evaluate_data('Environmental_parameters/length_of_service/Evaluation_data/dqn_evaluate_length_of_service_2_4.csv', evaluate_num, evaluate_reward, evaluate_sum_delay)

        # 打印关键指标
        print(f"评估轮次 {evaluate_num}: 总奖励={evaluate_reward:.2f}, 总时延={evaluate_sum_delay:.2f}, 步数={steps}")

    # 测试的奖励，时延，每一轮的奖励列表，采样长度
    return evaluate_reward, evaluate_sum_delay, rewards, steps


if __name__ == "__main__":
    import Environmental_parameters.length_of_service.length_of_service_2_4
    # torch.cuda.set_device(0)

    # 加载参数与环境
    args = Arguments.get_args()
    env_args = Environmental_parameters.length_of_service.length_of_service_2_4.get_args()
    env = ENV(env_args)

    # 启动训练
    train(env, args)
