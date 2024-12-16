"""
该文件用于执行模型的训练
"""
from Unnamed_Algorithm.Network import *
from Unnamed_Algorithm.Environment_Interaction import *
import torch.optim as optim

ITERATION_NUM = 50    # 训练轮数
GAMMA = 0.95            # 衰减率[0-1]
ACTOR_LR = 1e-4       # actor网络的学习率
CRITIC_LR = 1e-3      # critic网络的学习率
TAU = 0.05  # 目标网络软更新系数，用于软更新
MAX_DEPLOY_COUNT = 20  # 连续超过指定次数没有部署成功则认为当前节点无法部署
"""
经验回放：用于打破样本中的时间序列，但是如果神经网络中定义了lstm层，则需要谨慎使用
"""
class ReplayBuffer:
    def __init__(self, capacity):
        pass

class Agent:
    # 模型
    actor = None
    critic = None
    actor_target = None
    critic_target = None

    # 环境
    environment_interaction = None

    def __init__(self, environment_interaction=None):
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

        # 定义初始化的环境，若环境没有给出则自己定义
        if environment_interaction is None:
            self.environment_interaction = environment_interaction_ms_initial()
        else:
            self.environment_interaction = environment_interaction

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
        #     print(param.data)
            # print(param.grad)
    def train_ddpg(self):
        """
        执行训练
        :return: None
        """
        # 执行迭代，每一轮迭代结束以部署完成为标准
        for episode in range(ITERATION_NUM):
            state = initial_state() # 初始化状态
            self.environment_interaction.refresh()  # 刷新部署需求
            episode_count = 0   # 记录当前迭代的长度
            fail_count = 0  # 记录失败次数

            while True:
                # 产生动作，和下一个状态
                action_probabilities = self.actor(state)    # 由actor产生动作
                flag, next_state, reward = self.environment_interaction.get_next_state_and_reword(state,action_probabilities)

                # 部署结束
                if flag == 0:
                    # print(state)   # 查看状态
                    print("当前部署得到的延迟奖励为：", reward)   # 查看奖励
                    break

                # 部署未结束，继续生成下一个动作
                next_action_probabilities = self.actor(next_state)

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

                # 软更新目标网络
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                # 更新状态
                state = next_state.copy()
                # self.environment_interaction.analysis_state(state) # 检查状态
                episode_count += 1 # 记录次数

                ## 防死循环机制
                # 当出现一个服务在所有的服务器上都没法部署时，放弃部署该节点
                # 处理方式是记录当前部署失败的次数，超过指定次数后，放弃部署该服务
                fail_count = fail_count + 1 if reward == PUNISHMENT_DEPLOY_FAIL else 0
                if fail_count > MAX_DEPLOY_COUNT:
                    fail_count = 0
                    self.environment_interaction.pass_round()   # 跳过当前部署


            print(f"第 {episode} 次迭代执行了 {episode_count} 次训练")
            # self.show_parameters()

        print("训练完成")
    def get_deterministic_deployment(self, state=None):
        """
        对于给定的初始状态，生成一个部署方案
        用于测试目标网络部署选择方法
        :return: None
        """
        if state is None:
            state = initial_state()

        self.environment_interaction.refresh()

        self.environment_interaction.analysis_state(state)  # 测试专用

        num = 0
        fail_count = 0
        fail_ms_count = 0  # 记录没有被部署上的服务个数
        while True:
            action_probabilities = self.actor_target(state)
            flag, next_state, reward = self.environment_interaction.get_next_state_and_reword(state, action_probabilities)
            # print(reward, next_state)
            if flag == 0: break

            state = next_state

            num += 1

            ## 防死循环机制
            # 当出现一个服务在所有的服务器上都没法部署时，放弃部署该节点
            # 处理方式是记录当前部署失败的次数，超过指定次数后，放弃部署该服务
            fail_count = fail_count + 1 if reward == PUNISHMENT_DEPLOY_FAIL else 0
            if fail_count > MAX_DEPLOY_COUNT:
                fail_count = 0
                self.environment_interaction.pass_round()  # 跳过当前部署
                fail_ms_count += 1

        print(f"算法执行次数 {num} ,一共需要部署 {self.environment_interaction.sum_ms_aims} 个服务，其中有 {fail_ms_count} 个服务没有部署上",)
        self.environment_interaction.analysis_state(state)  # 测试专用
        return state    # 返回最终方案

    def save_model(self):
        """
        用于保存模型
        :return:None
        """
        torch.save(self.actor_target.state_dict(), "Model/actor_target_model.pth")
        torch.save(self.critic_target.state_dict(), "Model/critic_target_model.pth")
        print("模型已保存！")

    def load_model(self):
        """
        用于加载模型
        :return: None
        """
        self.actor_target.load_state_dict(torch.load("Model/actor_target_model.pth"))
        self.critic_target.load_state_dict(torch.load("Model/critic_target_model.pth"))

    def run(self):
        """
        按照全部部署完一次，算作迭代一次
        :return:
        """
        # 训练
        self.train_ddpg()
        res_state = self.get_deterministic_deployment()  # 最终结果
        self.save_model()


if __name__ == '__main__':

    agent = Agent()
    agent.run()