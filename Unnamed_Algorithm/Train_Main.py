"""
该文件用于执行模型的训练
"""
import json
import os.path
from collections import deque
from Unnamed_Algorithm.Network import *
from Unnamed_Algorithm.Environment_Interaction import *
import torch.optim as optim

ITERATION_NUM = 10000  # 训练轮数
GAMMA = 0.95  # 衰减率[0-1]
ACTOR_LR = 1e-4  # actor网络的学习率
CRITIC_LR = 1e-3  # critic网络的学习率
TAU = 0.05  # 目标网络软更新系数，用于软更新
MAX_DEPLOY_COUNT = 2 * MA_AIMS_NUM  # 连续超过指定次数没有部署成功则认为当前节点无法部署
BATCH_SIZE = 64  # 一个批次中的数据量大小（用于off policy）
CAPACITY = BATCH_SIZE * 100  # 经验回放池的大小
torch.autograd.set_detect_anomaly(True)

SAVE_COUNT = 100  # 每迭代几次就保存模型
"""
经验回放：用于打破样本中的时间序列，但是如果神经网络中定义了lstm层，则需要谨慎使用
"""
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_action):
        """
        添加样本
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 奖励
        :param next_state: 下一个状态
        :param next_action: 下一个动作
        :return: None
        """
        # detach() 可以断开梯度，从而将不带梯度的数据放入经验池
        self.buffer.append((state.copy(), action, reward, next_state.copy(), next_action))

    def sample(self, batch_size=BATCH_SIZE):
        """
        采样
        :param batch_size:
        :return:
        """
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


class Agent:
    # 模型
    actor = None
    critic = None
    actor_target = None
    critic_target = None

    # 环境
    environment_interaction = None

    # 经验回放池
    replay_buffer = None

    # 训练情况统计
    statistics = []

    def __init__(self):
        # 模型初始化
        self.actor = Actor(ma_aims_num=MA_AIMS_NUM, node_num=NODE_NUM)
        self.actor_target = Actor(ma_aims_num=MA_AIMS_NUM, node_num=NODE_NUM)
        self.critic = Critic(ma_aims_num=MA_AIMS_NUM, node_num=NODE_NUM)
        self.critic_target = Critic(ma_aims_num=MA_AIMS_NUM, node_num=NODE_NUM)

        # 统一权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器定义
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # 经验回放的定义
        self.replay_buffer = ReplayBuffer(CAPACITY)

        # 之前训练的记录
        self.old_episode_count = 0

        # 初始化环境
        self.init_environment()

    def init_environment(self):
        """
        初始化环境
        :return:
        """
        self.environment_interaction = environment_interaction_ms_initial()

    def show_parameters(self):
        """
        用于可视化模型参数
        :return: None
        """
        print("actor_target 模型参数：")
        for name, param in self.actor.named_parameters():
            print(param.data)
            # print(param.grad)

        # print("critic_target 模型参数：")
        # for name, param in self.actor_target.named_parameters():
        #     print(param.Data)
        # print(param.grad)

    def train_model(self, state, action_probabilities, reward, next_state, next_action_probabilities):
        """
        执行一次训练
        :param state: 当前状态
        :param action_probabilities: 当前状态的行动
        :param reward: 回报
        :param next_state: 下一个状态
        :param next_action_probabilities: 下一个状态的行动
        :return: None
        """
        ## 计算critic误差
        target_q_values = reward + GAMMA * self.critic_target(next_state, next_action_probabilities)
        current_q_values = self.critic(state, action_probabilities)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        ## 训练critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        ## 计算actor误差
        actor_loss = -self.critic(state, action_probabilities).mean()

        ## 训练Actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        ## 记录误差总和
        self.data['loss'] += actor_loss.item()

    def soft_update(self):
        """
        执行软更新
        :return:None
        """
        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def train_ddpg_on_policy(self):
        """
        执行on_policy 算法的训练
        :return: None
        """

        # 执行迭代，每一轮迭代结束以部署完成为标准
        for episode in range(1, ITERATION_NUM + 1):
            state = initial_state()  # 初始化状态
            self.environment_interaction.refresh()  # 更新镜像需求数
            episode_count = 0  # 记录当前迭代的长度
            fail_count = 0  # 记录失败次数
            sum_fail_count = 0  # 记录未能部署上的节点数目
            self.data = {    # 数据字典
                'episode': episode + self.old_episode_count,
                'sum_reward': 0,
                'loss': 0,
                'T': 0,
            }

            while True:
                # 产生动作，和下一个状态
                action_probabilities = self.actor(state)  # 由actor产生动作
                action, reward, next_state = self.sampling(state, episode_count)  # 采样

                # 部署结束
                if self.environment_interaction.is_it_over():
                    next_action_probabilities = action_probabilities
                    next_state = state.copy()
                else:  # 继续生成下一个动作
                    next_action_probabilities = self.actor(next_state)

                # 部署成功，执行训练
                # if reward != PUNISHMENT_DEPLOY_FAIL:
                # print("reward ==", reward)
                ## 训练
                # 执行训练模型的训练
                self.train_model(state, action_probabilities, reward, next_state, next_action_probabilities)
                # 执行软更新
                self.soft_update()

                ## 数据更新
                episode_count += 1  # 记录训练次数

                if reward != PUNISHMENT_DEPLOY_FAIL: self.data['sum_reward'] += reward
                # 部署结束退出循环
                if self.environment_interaction.is_it_over():
                    break

                state = next_state.copy() # 更新状态
                # self.environment_interaction.analysis_state(state) # 检查状态

                ## 防死循环机制
                # 当出现一个服务在所有的服务器上都没法部署时，放弃部署该节点
                # 处理方式是记录当前部署失败的次数，超过指定次数后，放弃部署该服务
                fail_count = fail_count + 1 if reward == PUNISHMENT_DEPLOY_FAIL else 0
                if fail_count > MAX_DEPLOY_COUNT:
                    fail_count = 0
                    sum_fail_count += 1
                    self.environment_interaction.pass_round()  # 跳过当前部署

            self.data['T'] = self.environment_interaction.get_T(state)
            self.statistics.append(self.data)   # 记录训练数据
            print(f"第 {episode} 次迭代执行了 {episode_count} 次训练, 当前部署得到的延迟为 {self.data['T']}，函数损失值loss为 {self.data['loss']} ，一共有 {self.environment_interaction.sum_ms_aims} 个待部署实例，其中有 {sum_fail_count} 个实例没有部署上")
            print(self.data)

            # 指定一段时间保存一次模型
            if episode % SAVE_COUNT == 0:
                self.save_model()
            # self.environment_interaction.analysis_state(state)

        print("训练完成")


    def off_train(self):
        """
        执行离线训练
        :return: None
        """
        # 当经验池中的样本数达到一个批次的量时才会能开始训练
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # 执行若干次训练模型的更新后，在执行软更新
        for state, action_probabilities, reward, next_state, next_action_probabilities in self.replay_buffer.sample():
            # print(state, action_probabilities, reward, next_state, next_action_probabilities)
            self.train_model(state, action_probabilities, reward, next_state, next_action_probabilities)
        self.soft_update()

    def train_ddpg_off_policy(self):
        """
        执行off_policy算法的训练
        :return: None
        """
        # 执行迭代，每一轮迭代结束以部署完成为标准
        for episode in range(ITERATION_NUM):
            state = initial_state()  # 初始化状态
            self.init_environment()  # 初始化环境（环境中已经自带了部署需求）
            episode_count = 0  # 记录当前迭代的长度
            fail_count = 0  # 记录失败次数
            sum_fail_count = 0  # 记录未能部署上的节点数目

            while True:
                # 产生动作，和下一个状态
                action_probabilities = self.actor(state)  # 由actor产生动作
                action, reward, next_state = self.sampling(state)  # 采样

                # 部署结束
                if self.environment_interaction.is_it_over():
                    next_action_probabilities = action_probabilities
                    next_state = state.copy()
                else:  # 继续生成下一个动作
                    next_action_probabilities = self.actor(next_state)

                # 执行训练模型的训练
                self.replay_buffer.add(state, action_probabilities, reward, next_state, next_action_probabilities)
                # 执行软更新
                self.off_train()

                episode_count += 1  # 记录次数

                # 部署结束退出循环
                if self.environment_interaction.is_it_over():
                    break

                # 更新状态
                state = next_state.copy()
                # self.environment_interaction.analysis_state(state) # 检查状态

                ## 防死循环机制
                # 当出现一个服务在所有的服务器上都没法部署时，放弃部署该节点
                # 处理方式是记录当前部署失败的次数，超过指定次数后，放弃部署该服务
                fail_count = fail_count + 1 if reward == PUNISHMENT_DEPLOY_FAIL else 0
                if fail_count > MAX_DEPLOY_COUNT:
                    fail_count = 0
                    sum_fail_count += 1
                    self.environment_interaction.pass_round()  # 跳过当前部署

            print(f"第 {episode} 次迭代执行了 {episode_count} 次训练, 当前部署得到的延迟为 {self.environment_interaction.get_T(state)}，延迟奖励为：{reward}，一共有 {self.environment_interaction.sum_ms_aims} 个待部署实例，其中有 {sum_fail_count} 个实例没有部署上")
            # self.environment_interaction.analysis_state(state)

        print("训练完成")

    def sampling(self, state, episode_count=1):
        """
         采样函数，根据当前状态给出行动，奖励，下一个状态
        :param state: state
        :param episode_count: 记录训练的轮数，用于判断是否为第一轮
        :return: (a_t, r_t, s_t+1, a_t+1)
        """
        action_probabilities = self.actor_target(state)  # 行动概率分布

        self.environment_interaction.index = index = self.environment_interaction.option_ms()  # 选择需要部署的类型

        action = self.environment_interaction.get_action(index, action_probabilities)  # 行动
        # print(index, action_probabilities[index] , action)
        next_state = self.environment_interaction.get_next_state(index, state, action)  # 状态

        if self.environment_interaction.is_it_sufficient(index, state, action):  # 可以分配
            if self.environment_interaction.is_it_over():   # 部署结束
                reward = self.environment_interaction.get_reward(0, state, next_state, episode_count)
            else:                       # 部署未结束
                reward = self.environment_interaction.get_reward(1, state, next_state, episode_count)
        else:  # 不能分配
            reward = self.environment_interaction.get_reward(-1,episode_count=episode_count)

        return action, reward, next_state

    def get_deterministic_deployment(self, state=None, environment_interaction=None):
        """
        对于给定的初始状态，生成一个部署方案
        用于测试目标网络部署选择方法
        """

        if state is None:
            state = initial_state()

        if environment_interaction is None:
            self.init_environment()  # 初始化环境（环境中已经自带了部署需求）
        else:
            self.environment_interaction = environment_interaction

        self.environment_interaction.analysis_state(state, flag=True)  # 测试专用

        num = 0
        fail_count = 0
        fail_ms_count = 0  # 记录没有被部署上的服务个数

        # 持续迭代，直到部署结束
        while not self.environment_interaction.is_it_over():
            # 采样
            action, reward, next_state = self.sampling(state)

            state = next_state  # 状态更新
            num += 1  # 记录迭代次数
            # self.environment_interaction.analysis_state(state)  # 查看状态

            ## 防死循环机制
            # 当出现一个服务在所有的服务器上都没法部署时，放弃部署该节点
            # 处理方式是记录当前部署失败的次数，超过指定次数后，放弃部署该服务
            fail_count = fail_count + 1 if reward == PUNISHMENT_DEPLOY_FAIL else 0
            if fail_count > MAX_DEPLOY_COUNT:
                fail_count = 0
                self.environment_interaction.pass_round()  # 跳过当前部署
                fail_ms_count += 1

        print(f"算法执行次数 {num} ,时延{self.environment_interaction.get_T(state)},一共需要部署 {self.environment_interaction.sum_ms_aims} 个服务，其中有 {fail_ms_count} 个服务没有部署上", )
        self.environment_interaction.analysis_state(state)  # 测试专用
        return state  # 返回最终方案

    def save_model(self):
        """
        用于保存模型，由于不同模型接受的参数规模不一样（这是由于服务器数量，微服务类型数量等因素导致的），所以这里按照输入的类型进行命名
        :return:None
        """
        # 保存模型
        torch.save(self.actor_target.state_dict(), f"Model/actor_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}.pth")
        torch.save(self.critic_target.state_dict(), f"Model/critic_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}.pth")

        # 保存数据
        with open('Data/statistics.json', 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, ensure_ascii=False, indent=4)

        print("模型已保存！")

    def load_model(self):
        """
        当模型存在时，用于加载模型
        :return: None
        """
        if os.path.exists('Data/statistics.json'):
            with open('Data/statistics.json', 'r', encoding='utf-8') as f:
                self.statistics = json.load(f)
                self.old_episode_count = len(self.statistics)
                for data in self.statistics:
                     self.environment_interaction.T_min = min(self.environment_interaction.T_min, data['T'])

        actor_url = f"Model/actor_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}.pth"
        critic_url = f"Model/critic_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}.pth"

        if os.path.exists(actor_url) and os.path.exists(critic_url):
            self.actor_target.load_state_dict(torch.load(actor_url))
            self.critic_target.load_state_dict(torch.load(critic_url))

            # 网络复制
            self.actor.load_state_dict(self.actor_target.state_dict())
            self.critic.load_state_dict(self.critic_target.state_dict())

    def run(self):
        """
        按照全部部署完一次，算作迭代一次
        :return:
        """
        # 读取模型
        self.load_model()
        # 训练
        self.train_ddpg_on_policy()
        # self.train_ddpg_off_policy()
        # res_state = self.get_deterministic_deployment()  # 最终结果

        # 保存模型
        # self.save_model()


if __name__ == '__main__':
    agent = Agent()
    # print(agent.actor_target(state))
    agent.run()
    # print(agent.actor_target(state))

