"""
该脚本用于实现微服务的环境交互工作
"""
from DAC.Network import *

torch.manual_seed(0)  # 随机数种子
PUNISHMENT_DEPLOY_FAIL = -1  # 部署失败的惩罚

class Environment_Interaction:
    # 奖励计算的参数
    C1, C2, C3, C4 = 0.1, 1, 0.1, 0.001
    # 全局变量，历史最小时延
    T_min = 1e+10

    def option_ms(self):
        """
        选择一个微服务进行分配
        根据当前所需的实例数情况，返回最高需求量的那个
        返回-1表示已经全部部署完毕
        :return:MaxIndex
        """
        index = np.argmax(self.ms_image)
        if self.ms_image[index] <= 0:
            return -1
        return index

    def is_it_sufficient(self, index, state, action):
        """
        判断某一个服务资源是否够分
        :param index: 选择的微服务类型
        :param state: 状态
        :param action: 行动，即选择部署的节点
        :return: bool 表示是否分配成功
        """

        # 分离得到资源情况
        resource = get_resource(state)

        # 需要消耗的资源情况
        cpu = self.ms_aims[index].get_cpu()
        gpu = self.ms_aims[index].get_gpu()
        memory = self.ms_aims[index].get_memory()

        # 当前的资源状况
        # print(resource)
        now_cpu = resource[NODE_NUM: NODE_NUM * 2][action]
        now_gpu = resource[NODE_NUM * 3: NODE_NUM * 4][action]
        now_memory = resource[NODE_NUM * 5:][action]

        return now_cpu >= cpu and now_memory >= memory and now_gpu >= gpu



    def allocate_resources(self, index, state, action):
        """
        给某一个服务分配资源，若不够用则返回-1
        :param index: 选择的微服务类型
        :param state: 状态
        :param action: 行动，即选择部署的节点
        :return: bool 表示是否分配成功
        """
        if not self.is_it_sufficient(index, state, action):
            # # 分离得到资源情况
            # resource = get_resource(state)
            #
            # # 需要消耗的资源情况
            # cpu = self.ms_aims[index].get_cpu()
            # gpu = self.ms_aims[index].get_gpu()
            # memory = self.ms_aims[index].get_memory()
            #
            # now_cpu = resource[NODE_NUM: NODE_NUM * 2][action]
            # now_gpu = resource[NODE_NUM * 3: NODE_NUM * 4][action]
            # now_memory = resource[NODE_NUM * 5:][action]
            # print(f'资源不够不能分配！当前CPU：{now_cpu}、GPU：{now_gpu}、内存：{now_memory}，需要的资源：CPU：{cpu}、GPU：{gpu}、内存：{memory}')
            return False

        # 分离得到部署情况和资源情况
        deploy = get_deploy(state)
        resource = get_resource(state)

        # 需要消耗的资源情况
        cpu = self.ms_aims[index].get_cpu()
        gpu = self.ms_aims[index].get_gpu()
        memory = self.ms_aims[index].get_memory()

        # 可以分配资源，开始分配
        ## 分配实例数
        # print(state)
        self.ms_image[index] -= 1
        deploy[index][action] += 1
        ## 配平相应的资源
        # cpu分配
        resource[NODE_NUM: NODE_NUM * 2][action] -= cpu
        resource[:NODE_NUM][action] += cpu
        # gpu分配
        resource[NODE_NUM * 3: NODE_NUM * 4][action] -= gpu
        resource[NODE_NUM * 2: NODE_NUM * 3][action] += gpu
        # 内存分配
        resource[NODE_NUM * 5:][action] -= memory
        resource[NODE_NUM * 4: NODE_NUM * 5][action] += memory

        return True

    def is_it_over(self):
        """
        用于判断部署是否已经结束
        :return: None
        """
        return self.option_ms() == -1

    def get_action(self, index, action_probabilities):
        """
        对指定的服务类型按照概率选择一个行动
        :param index: 服务类型
        :param action_probabilities: 行动概率分布
        :return: action
        """
        action = torch.multinomial(action_probabilities[index], num_samples=1).item()
        return action

    def get_next_state(self,index, state, action):
        """
        依照概率选择下一个状态
        :param state: index
        :param state: state
        :param action: action概率分布
        :return: None
        """
        # 状态初始化
        next_state = state.copy()

        # 对next_state 进行分配，注意这里的next_state 传入的是引用变量对象
        self.allocate_resources(index, next_state, action)

        return next_state

    # def get_next_state_and_reword(self, state, action_probabilities):
    #     """
    #     根据当前状态和行动列表执行下一步行动得到新的状态
    #     标识符：部署成功返回1，部署失败返回 -1，部署结束返回 0
    #     :param state:
    #     :param action_probabilities:
    #     :return:部署状态标识符，状态，奖励
    #     """
    #     # 初始化
    #     next_state = state.copy()
    #
    #     ## 找到要部署的服务
    #     self.index = self.option_ms()
    #     if self.index == -1:
    #         # print("已部署完成")
    #         return 0, next_state, self.get_reward(0, state)
    #
    #     # print("待分配实例数情况：", self.ms_image)
    #     # print(f"选择第{self.index}个服务进行部署")
    #
    #     # 找到行动，即按指定概率随机选择一个服务节点
    #     action = torch.multinomial(action_probabilities[self.index], num_samples=1).item()
    #     # print(f"微服务{self.index},的需求为CPU:{self.ms_aims[self.index].get_cpu()},GPU:{self.ms_aims[self.index].get_gpu()},内存:{self.ms_aims[self.index].get_memory()}")
    #     # print("当前的actor产生的行动（每个微服务选择每一个服务器节点部署的概率分布）如下：\n", action_probabilities)
    #     # print(f"根据概率{action_probabilities[self.index]}选择了行动{action}，作为部署节点")
    #     ## 资源分配
    #
    #     # 分配失败
    #     if not self.allocate_resources(self.index, next_state, action):
    #         # print("分配失败")
    #         return -1, next_state, self.get_reward(-1)
    #
    #     # 分配成功，返回下一个状态，失败返回的状态和原来一样
    #     return 1, next_state, self.get_reward(1)

    def get_T(self, state):
        """
        根据状态计算时延
        :param state: state
        :return: T
        """
        deploy = get_deploy(state)
        rout = get_each_request_rout(deploy)
        T = cal_total_delay(deploy, rout)
        return T

    def get_reward(self, flag, state=None, next_state=None, episode_count=1):
        """
        针对部署的情况给予一定的奖励，（该函数待修改！）
        :param flag: 一个整数标识符，用于表示当前部署情况，0表示部署完成，-1表示部署失败，1表示分配成功
        :param state: 状态，可以用于再部署完成的情况下计算时延
        :param next_state: 下一个状态，可以用于再部署完成的情况下计算时延
        :param episode_count: 部署的轮数
        :return: 一个整数，用于表示rework
        """
        if flag != -1:  # 部署成功
            T_next = self.get_T(next_state)
            # print(T_min, T_next)
            if episode_count == 0:  # 部署第一个节点
                T = self.get_T(state)
                r = self.C1 * (T - T_next)
            elif T_next < self.T_min + 1/self.C3:   # 产生了更好的方案
                r = self.C2 + self.C3*(self.T_min - T_next)
            else:
                r = self.C4 * (self.T_min - T_next)

            self.T_min = min(self.T_min, T_next)    # 更新最小时延
            return r
        else:  # 部署失败, 基于惩罚
            return PUNISHMENT_DEPLOY_FAIL

    def pass_round(self):
        """
        当节点无法部署时使用，跳过当前一轮服务部署
        :return: None
        """
        self.ms_image[self.index] -= 1

    def refresh(self):
        """
        刷新，即重新分配实例数
        :return: None
        """
        self.ms_image = self.old_ms_image.copy()

    def analysis_state(self, state, index=None, flag=False):
        """
        用于分析，某状态
        :param state:
        :param flag:
        :param index: 选择部署的微服务
        :return:
        """
        print("*" * 30, self.count, "*" * 30)
        self.count += 1

        if flag:
            print("每一种微服务资源所需情况如下：")
            for i, s in enumerate(self.ms_aims):
                print(f"服务{i}，CPU: {s.get_cpu()}, GPU: {s.get_gpu()},内存: {s.get_memory()}")

        if index:
            s = self.ms_aims[index]
            print(f"已结束对服务 {index} 的部署: CPU: {s.get_cpu()}, GPU: {s.get_gpu()},内存: {s.get_memory()}")

        print("当前待分配实例数：\n", self.ms_image)

        print("当前的部署情况：")
        deploy = get_deploy(state).T
        print("服务类型\t \t", '\t \t'.join([f"{i}" for i in range(MA_AIMS_NUM)]))
        for i, node_deployment in enumerate(deploy):
            print(f"服务器{i}\t|\t", '\t|\t'.join([str(int(x)) for x in node_deployment]))

        print("服务器资源状态（已用|剩余|总量）：")
        deploy = get_resource(state)
        CUP = deploy[:NODE_NUM * 2]
        GUP = deploy[NODE_NUM * 2:NODE_NUM * 4]
        Memory = deploy[NODE_NUM * 4:]
        for i in range(NODE_NUM):
            print(
                f"服务器{i}: \tCPU:{CUP[i]} | {CUP[i + NODE_NUM]} | {CUP[i] + CUP[i + NODE_NUM]} \tGPU:{GUP[i]} | {GUP[i + NODE_NUM]} | {GUP[i] + GUP[i + NODE_NUM]} \t内存:{Memory[i]} | {Memory[i + NODE_NUM]} | {Memory[i + NODE_NUM] + Memory[i]}")
        print()



    def __init__(self, ms_image, all_ms):
        """
        初始化时，需要给实例数
        :param ms_image: 实例数镜像
        """
        # 初始化资源镜像
        self.old_ms_image = ms_image.copy()
        self.ms_image = self.old_ms_image.copy()
        self.sum_ms_aims = int(sum(self.old_ms_image))   # 待部署的服务总数
        # 初始化服务
        self.ms_aims = all_ms

        # 计数器
        self.count = 0


def environment_interaction_ms_initial():
    """
    创造一个环境交互的初始化对象
    :return: None
    """
    # 制作一个待分配实例数，这里定义全局变量，使得函数外的全局变量可以受到影响
    global all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, bandwidth, data, connected_lines, graph
    all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, bandwidth, data = environment_initialization()
    # ms_image = get_ms_image(all_ms_alpha, users, user_list, marker)
    ms_image = get_ms_image()
    # 初始化环境
    env = Environment_Interaction(ms_image, all_ms)
    return env


if __name__ == '__main__':
    # 制作一个待分配实例数
    # all_ms, all_ms_alpha, node_list, users, user_list, service_lamda, marker, bandwidth, Data, graph, connected_lines = environment_initialization()
    # ms_image = get_ms_image(all_ms_alpha, users, user_list, marker)
    # ms_image = get_ms_image()
    #
    # # 初始化环境
    # # print(ms_image, type(ms_image))
    # env = Environment_Interaction(ms_image, all_ms)
    # # print(env.option_ms())
    #
    # # 生成一个初始状态
    # state = initial_state()
    # env.analysis_state(state)
    # # 模型
    # actor = Actor(MA_AIMS_NUM, NODE_NUM)
    # critic = Critic(MA_AIMS_NUM, NODE_NUM)
    #
    # # 测试部分
    # print("每一种微服务资源所需情况如下：")
    # for i, s in enumerate(all_ms):
    #     print(f"服务{i}，所需资源:CPU:{s.get_cpu()}, GPU:{s.get_gpu()},内存：{s.get_memory()}")
    # for _ in range(5):
    #     # print("当前状态:\n", state)
    #     print("当前部署情况:\n", get_deploy(state))
    #     print("当前资源情况:\n", get_resource(state))
    #     action_list = actor(state)
    #     _, next_state, r = env.get_next_state_and_reword(state, action_list)
    #     state = next_state
    #     print("部署奖励：", r)
    global all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, bandwidth, data, connected_lines, graph
    for _ in range(3):

        all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, bandwidth, data = environment_initialization()
        connected_lines, graph = connect_nodes_within_range(node_list, initial_range=10)
        ms_image = get_ms_image()
        print(ms_image)
        state = initial_state()
        # print(state)