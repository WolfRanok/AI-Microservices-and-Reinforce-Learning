"""
该脚本用于实现微服务的环境交互工作
"""
import numpy as np

from Agent import *

torch.manual_seed(0)  # 随机数种子
PUNISHMENT_DEPLOY_FAIL = -1  # 部署失败的惩罚

class Environment_Interaction:
    # 奖励计算的参数
    C1, C2, C3, C4 = 0.1, 1, 0.1, 0.001
    # 全局变量，历史最小时延
    T_min = 1e+10
    def __init__(self, ms_image, all_ms):
        """
        初始化时，需要给实例数
        :param ms_image: 实例数镜像
        """
        # 初始化资源镜像
        self.org_ms_image = ms_image.copy()
        self.each_image = np.sum(ms_image, -1)
        self.each_request_image = self.each_image.copy()
        ms_image = np.reshape(ms_image, (USER_NUM * MA_AIMS_NUM))
        self.old_ms_image = ms_image.copy()
        self.ms_image = self.old_ms_image.copy()
        self.sum_ms_aims = int(sum(self.old_ms_image))   # 待部署的服务总数
        # 初始化服务
        self.ms_aims = all_ms
        self.adv = np.full((sum(ms_image).astype(int),), 0)
        self.T_list = np.full((sum(ms_image).astype(int),), 0.0)
        self.T_min_list = np.full((sum(ms_image).astype(int),), cal_total_delay(get_deploy(initial_state()),get_each_request_rout(get_deploy(initial_state()))))
        self.dep_ms_count = -1 # 表示尝试部署过的微服务实例数量-1
        self.ms_deploy_idx = get_ms_deploy_order()
        self.each_ms_num = self.cal_each_ms_num()
        self.now_dep_ms_class_num = 0
        # 计数器
        self.count = 0
    def cal_each_ms_num(self):
        each_ms_num = np.zeros((MA_AIMS_NUM))
        for i in range(MA_AIMS_NUM):
            for j in range(USER_NUM):
                each_ms_num[i] += self.org_ms_image[j][i]
        return each_ms_num
    def option_ms(self):
        """
        选择一个微服务进行分配，采用按链部署的策略
        返回-1表示已经全部部署完毕
        :return:MaxIndex
        """
        # index = np.argmax(self.ms_image)
        dep_is_over = True
        index = -1
        for i in self.ms_deploy_idx:
            if self.ms_image[i]!=0:
                index = i % (MA_AIMS_NUM)
                dep_is_over = False
                break
        if dep_is_over:
            return -1,
        return index
    def get_action(self, index, action_probabilities):
        """
        对指定的服务类型按照概率选择一个行动
        :param index: 服务类型
        :param action_probabilities: 行动概率分布
        :return: action
        """
        p = random.uniform(0,1)
        e_p = max(0.8,torch.argmax(action_probabilities).item())
        if p < e_p:
            action = torch.argmax(action_probabilities).item()
        else:
            action = torch.multinomial(action_probabilities, num_samples=1).item()
        # action = torch.multinomial(action_probabilities, num_samples=1).item()

        return action
    def get_next_state(self,index, state, action):
        """
        依照概率选择下一个状态
        :param state: index 微服务编号，AI微服务接着基础微服务后面继续编号
        :param state: state
        :param action: action概率分布
        :return: None
        """
        # 状态初始化
        next_state = state.copy()
        # 对next_state 进行分配，注意这里的next_state 传入的是引用变量对象
        # print(self.now_dep_ms_class_num)
        if self.is_it_sufficient(index, state, action) and sum(self.ms_image) != 0:
            # 可以分配资源，开始分配
            ## 分配实例数
            # print(state)
            # 更新state中实时记录的ms_image
            # self.ms_image[self.ms_deploy_idx[self.now_dep_ms_class_num%len(self.ms_deploy_idx)]] -= 1
            # print(self.ms_image)
            this_ms_index = index
            for i in range(USER_NUM):
                if self.ms_image[index + i * MA_AIMS_NUM] != 0:
                    self.each_request_image[i] -= 1
                    this_ms_index = index + i * MA_AIMS_NUM
                    break
            self.ms_image[this_ms_index] -= 1

            # 改变服务部署方案与资源剩余情况
            next_state = updata_state(state, action, index)
            # start = MA_AIMS_NUM*NODE_NUM+ 3*2*NODE_NUM
            # state[start:start+USER_NUM*MA_AIMS_NUM] = self.ms_image.copy()
        else:
            # 此时，微服务实例部署失败，但是还是需要更新state中的ms_image，但不更新部署方案和资源占用情况
            # 更新state中实时记录的ms_image
            # self.ms_image[self.ms_deploy_idx[self.now_dep_ms_class_num%len(self.ms_deploy_idx)]] -= 1
            # print(self.ms_image)

            this_ms_index = index
            for i in range(USER_NUM):
                if self.ms_image[index + i * MA_AIMS_NUM] != 0:
                    self.each_request_image[i] -= 1
                    this_ms_index = index + i * MA_AIMS_NUM
                    break
            self.ms_image[this_ms_index] -= 1
            # start = MA_AIMS_NUM * NODE_NUM + 3 * 2 * NODE_NUM
            # state[start:start + USER_NUM * MA_AIMS_NUM] = self.ms_image.copy()

        # 更新一下下一次需要部署的微服务实例信息以及所属的用户请求信息
        if sum(self.ms_image) != 0:
            start = MA_AIMS_NUM * NODE_NUM + 3 * 2 * NODE_NUM
            cur_ms_information = state[start:(start + 4)]
            cur_request_information = state[(start + 4):(start + 8)]
            # print(cur_ms_information)
            # print(cur_request_information)
            # print(users[int(self.ms_deploy_idx[self.now_dep_ms_class_num]/MA_AIMS_NUM)].id)
            # print(users[int(self.ms_deploy_idx[self.now_dep_ms_class_num]/MA_AIMS_NUM)].get_lamda())
            # print(self.ms_image[self.ms_deploy_idx[self.now_dep_ms_class_num+7]])
            if self.ms_image[self.ms_deploy_idx[self.now_dep_ms_class_num%len(self.ms_deploy_idx)]] == 0:
                self.now_dep_ms_class_num += 1
                next_ms = all_ms[self.ms_deploy_idx[self.now_dep_ms_class_num%len(self.ms_deploy_idx)] % MA_AIMS_NUM]
                cur_ms_information[0] = next_ms.id
                cur_ms_information[1] = next_ms.get_cpu()
                cur_ms_information[2] = next_ms.get_gpu()
                cur_ms_information[3] = next_ms.get_memory()
                next_request = users[int(self.ms_deploy_idx[self.now_dep_ms_class_num%len(self.ms_deploy_idx)] / MA_AIMS_NUM)]
                cur_request_information[0] = next_request.id
                cur_request_information[1] = next_request.get_lamda()
                next_request_x, next_request_y = next_request.get_location()
                cur_request_information[2] = next_request_x
                cur_request_information[3] = next_request_y
            state[start:(start + 4)] = cur_ms_information
            state[(start + 4):(start + 8)] = cur_request_information
        return next_state

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
    # 计算每一个服务请求在state下的时延
    def get_each_T(self, state):
        """
        根据状态计算时延
        :param state: state
        :return: T
        """
        deploy = get_deploy(state)
        rout = get_each_request_rout(deploy)
        T = cal_each_service_delay(deploy, rout)
        return T
    def get_reward(self, flag, ms_index, state=None, next_state=None, episode_count=1):
        """
        针对部署的情况给予一定的奖励，（该函数待修改！）
        :param flag: 一个整数标识符，用于表示当前部署情况，0表示部署完成，-1表示部署失败，1表示分配成功
        :param state: 状态，可以用于再部署完成的情况下计算时延
        :param next_state: 下一个状态，可以用于再部署完成的情况下计算时延
        :param episode_count: 部署的轮数
        :return: 一个整数，用于表示rework
        """
        T_next = self.get_T(next_state)
        T = self.get_T(state)
        loadb = cal_load_balance(next_state)
        _, T_max = cal_each_D_max()
        idx = self.dep_ms_count % self.T_min_list.size
        # print(idx)
        # 奖励构成一：是否部署成功
        if isinstance(all_ms[ms_index], MS):
            if flag != -1:  # 部署成功
                r = 1 / self.each_ms_num[ms_index]
            else:  # 部署失败, 基于惩罚
                r = 0
        else:
            if flag != -1:  # 部署成功
                r = 10/self.each_ms_num[ms_index]
            else:  # 部署失败, 基于惩罚
                r = -5

        # if self.T_min_list[idx] >= T_next and self.T_min_list[idx] <= T_next+1:
        #     r += 1 - loadb
        # else:
        #     r += 1 + 0.1 * (self.T_min_list[idx] - T_next) - loadb
        if idx == self.T_min_list.size-1:
            print("yes")
            if self.T_min_list[-1] == T_next:
                r += 1 - loadb
            else:
                r += 1 + 0.1 * (self.T_min_list[-1] - T_next) - loadb
        self.T_list[idx] = T_next
        if self.T_min_list[idx] > T_next:
            self.T_min_list[idx] = T_next
        return r

    def pass_round(self, this_index,state):
        """
        当节点无法部署时使用，跳过当前一轮服务部署
        :return: None
        """
        # 更新微服务实例部署数量
        this_ms_index = this_index
        for i in range(USER_NUM):
            if self.ms_image[this_index + i * MA_AIMS_NUM] != 0:
                this_ms_index = this_index + i * MA_AIMS_NUM
                break
        self.ms_image[this_ms_index] -= 1
        # 更新状态中的微服务实例部署数量
        start = MA_AIMS_NUM * NODE_NUM + 3 * 2 * NODE_NUM
        state[start:start + USER_NUM * MA_AIMS_NUM] = self.ms_image.copy()
        # self.dep_ms_count += 1
        # print(f"已部署的微服务实例数量(不论成功与否){self.dep_ms_count}")
        # 记录当前状态的响应时延以及最低响应时延
        idx = self.dep_ms_count % self.T_min_list.size
        T = self.get_T(state)
        self.T_list[idx] = T
        if self.T_min_list[idx] > T:
            self.T_min_list[idx] = T
    def is_it_over(self):
        """
        用于判断部署是否已经结束
        :return: None
        """
        return sum(self.ms_image)==0
    # 判断当前部署的请求链中的微服务是否都部署完成
    def this_reuqest_is_over(self):
        for idx in range(USER_NUM-1):
            if self.each_request_image[idx]==0 and self.each_request_image[idx+1]== self.each_image[idx+1]:
                return True
            return False
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
def environment_interaction_ms_initial():
    """
    创造一个环境交互的初始化对象
    :return: None
    """
    # 制作一个待分配实例数，这里定义全局变量，使得函数外的全局变量可以受到影响
    global all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, bandwidth, data, connected_lines, graph
    all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, bandwidth, data, request_data = environment_initialization()
    # ms_image = get_ms_image(all_ms_alpha, users, user_list, marker)
    ms_image = get_each_req_ms_image()
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
    e = environment_interaction_ms_initial()

    print(e.ms_deploy_idx)
    print(e.ms_image)
    state = initial_state()
    e.get_next_state(1,state,1)
    for u in users:
        print(f"请求到达率{u.lamda}", end=' ')
        print(' ')
        print("服务请求：", end=' ')
        for i in requests.get(u):
            print(i.id, end=' ')
        print(' ')
        for i in marker.get(u):
            print(i, end=' ')
        print(' ')
