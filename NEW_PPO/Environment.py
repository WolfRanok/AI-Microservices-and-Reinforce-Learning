import random

import math
import numpy as np
import torch
from torch_geometric.data import Data,Dataset, DataLoader,Batch


import Arguments
import network
from Object_Parameter_DEF import *

random.seed(3)
np.random.seed(3)
class ENV():
    def __init__(self, env_args):
        self.MS_NUM = env_args.ms_num
        self.AIMS_NUM = env_args.aims_num
        self.USER_NUM = env_args.user_num
        self.NODE_NUM = env_args.node_num
        self.RESOURCE = env_args.resource
        # self.MAX_LENGTH = env_args.max_length_of_service
        self.MS_AIMS_NUM = self.MS_NUM + self.AIMS_NUM
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.ms_list = self.ms_initial() # 基础微服务
        self.aims_list = self.aims_initial() # AI微服务
        self.all_ms_list = [item for sublist in [self.ms_list, self.aims_list] for item in sublist] # 微服务集合，AI微服务在最后
        self.user_list = self.user_initial(env_args) # 用户，即服务
        self.node_list = self.edge_initial(env_args.gpu_of_node_num) # 服务器集合
        self.connected_lines, self.graph = self.connect_nodes_within_range() # 网络拓扑

        self.Ms_types_global_arrival_rates = self.get_ms_global_arrival_rates() # 微服务实例的全局到达率
        self.ms_first_node = self.get_ms_arrival_first_node() # 微服务实例的入口服务器集合
        self.user_first_arrival_node_list = self.get_user_first_arrival_node() # 服务的入口服务器

        self.Ms_types_demand = [0 for _ in range(self.MS_AIMS_NUM)]  # 每种微服务实例实际在环境中所需部署的数量
        self.Ms_types_instances = self.get_Ms_types_instances() # 需要部署的微服务实例
        self.Ms_instance_sum = len(self.Ms_types_instances)  # 将要被部署在环境中的微服务实例总数

        self.GNN_ms_dependency = self.get_ms_graph() # 微服务调用图
        self.min_episode_sum_delay = 80  # 训练轮次中出现的最短总时延
        self.last_episode_sum_delay = 80  # 上一训练轮次的总时延

        # 每次部署都会变的变量
        self.MS_instances_deployed_on_server = [[0 for _ in range(self.NODE_NUM)] for _ in range(self.MS_AIMS_NUM)] # 网络中的部署方案
        self.CPU_resource_on_service = [node.get_cpu() for node in self.node_list]
        self.GPU_resource_on_service = [node.get_gpu() for node in self.node_list]
        self.Memory_resource_on_service = [node.get_memory() for node in self.node_list]
        self.init_deployment_and_resource_solution() # 初始化部署方案
        self.Service_rout_table = self.get_each_request_rout()  # 网络中服务的路由方案
        self.current_ms_index = 0  # 当前正在部署的微服务实例在实例集合中的位置
        self.last_step_sum_delay = self.get_total_delay()   # 时延计算

        self.global_step = 0    # 训练步数
    def ms_initial(self):
        ms_list = []
        for i in range(self.MS_NUM):
            ms_list.append(MS(i))
        return ms_list
    def aims_initial(self):
        aims_list = []
        for i in range(self.AIMS_NUM):
            aims_list.append(AIMS(i))
        return aims_list
    def user_initial(self, env_args):
        x_node = np.loadtxt(open("users.CSV"), delimiter=",", skiprows=1, usecols=[1])
        y_node = np.loadtxt(open("users.CSV"), delimiter=",", skiprows=1, usecols=[2])
        user_list = []
        for i in range(self.USER_NUM):
            user_list.append(USER(i, x_node[(i - 1)%20], y_node[(i - 1)%20], self.ms_list, self.aims_list,
                                  env_args.min_arrival_rate, env_args.max_arrival_rate, env_args.AI_service_num,
                                  env_args.min_length_of_f_service, env_args.max_length_of_f_service,
                                  env_args.min_length_of_f_service_in_ai, env_args.max_length_of_f_service_in_ai,
                                  env_args.min_length_of_ai_service_in_ai, env_args.max_length_of_ai_service_in_ai))
        return user_list
    def edge_initial(self, num_of_gpu):
        # edge_node_list = []
        # for i in range(self.NODE_NUM):
        #     edge_node_list.append(EDGE_NODE(i))
        index = np.random.choice(range(0,20),size=10,replace=False)
        x_node = np.loadtxt(open("edge_node.CSV"), delimiter=",", skiprows=1, usecols=[1])
        x_node = x_node[:self.NODE_NUM]
        y_node = np.loadtxt(open("edge_node.CSV"), delimiter=",", skiprows=1, usecols=[2])
        y_node = y_node[:self.NODE_NUM]
        x_mean = sum(x_node) / len(x_node)
        y_mean = sum(y_node) / len(y_node)
        edge_node_list = []
        for i in range(self.NODE_NUM):
            gpu = 0
            d = math.sqrt((x_mean - x_node[i]) ** 2 + (y_mean - y_node[i]) ** 2) * 111
            # print(f"中心到服务器{i}的距离{d}")
            if d < 0.5:
                # gpu = random.randint(5,10)
                # gpu = random.randint(10, 15)
                gpu = num_of_gpu
            edge_node_list.append(EDGE_NODE(i, x_node[i - 1], y_node[i - 1], gpu))
        return edge_node_list
    def cal_dis(self, node1, node2):
        disx = (node1.x - node2.x) ** 2
        disy = (node1.y - node2.y) ** 2
        dis = math.sqrt(disx + disy)
        return dis
    def cal_dis_user_node(self, user, node):
        disx = (node.x - user.x) ** 2
        disy = (node.y - user.y) ** 2
        dis = math.sqrt(disx + disy)
        return dis
    def connect_nodes_within_range(self):
        """
        确保节点完全连通，处理不连通的情况。
        初始距离设为10，每次扩大范围增量为1
        """
        connected_lines = []
        V = [[0] * self.NODE_NUM for _ in range(self.NODE_NUM)]
        for i in range(self.NODE_NUM):
            for j in range(self.NODE_NUM):
                if i == j:
                    V[i][j] = 1
                if i != j:
                    dis = self.cal_dis(self.node_list[i], self.node_list[j]) * 111
                    if dis < 5:
                        V[i][j] = 1
                        connected_lines.append((self.node_list[i].id, self.node_list[j].id))
        return connected_lines, V

    def get_ms_arrival_first_node(self):
        node_priority = sorted(range(self.NODE_NUM), key=lambda k: self.node_list[k].bandwidth, reverse=True)
        cloud_node = node_priority[:3]
        edge_node = node_priority[3:]
        all_ms_priority = sorted(range(self.MS_AIMS_NUM), key=lambda k: self.Ms_types_global_arrival_rates[k], reverse=True)
        # 微服务依赖
        dependency = np.zeros((self.MS_AIMS_NUM, self.MS_AIMS_NUM), dtype=int)
        for i in range(self.MS_AIMS_NUM):
            dependency[i][i] = 1
        for user in self.user_list:
            request = user.request_chain
            pre_ms = request[0].id
            for ms_item in request[1:]:
                if isinstance(ms_item, MS):
                    dependency[pre_ms][ms_item.id] += 1
                    dependency[ms_item.id][pre_ms] += 1
                    pre_ms = ms_item.id
                else:
                    dependency[pre_ms][ms_item.id + self.MS_NUM] += 1
                    dependency[ms_item.id + self.MS_NUM][pre_ms] += 1
                    pre_ms = ms_item.id + self.MS_NUM

        ms_priority = []
        aims_priority = []
        for idx in range(self.MS_AIMS_NUM):
            if all_ms_priority[idx] < self.MS_NUM:
                ms_priority.append(all_ms_priority[idx])
            else:
                aims_priority.append(all_ms_priority[idx])
        ms_prefer_node = {}
        for idx in range(len(aims_priority)):
            ms_prefer_node[self.all_ms_list[aims_priority[idx]]] = cloud_node[idx]
            max_dependency = 0
            cur_ms = -1
            for ms in ms_priority:
                if max_dependency<dependency[aims_priority[idx]][ms]:
                    cur_ms = ms
            if cur_ms!=-1:
                ms_prefer_node[self.all_ms_list[cur_ms]] = cloud_node[idx]
                ms_priority.remove(cur_ms)
        for idx in range(len(ms_priority)):
            if idx < len(edge_node):
                ms_prefer_node[self.all_ms_list[ms_priority[idx]]] = edge_node[idx]
            else:
                ms_d = []
                for ms in ms_priority:
                    ms_d.append(dependency[ms][ms_priority[idx]])
                max_d = 0
                cur_node = -1
                for index in range(len(ms_priority)):
                    if max_d<ms_d[index]:
                        max_d=ms_d[index]
                        cur_node = ms_prefer_node.get(self.all_ms_list[ms_priority[index]])
                if cur_node!=-1:
                    ms_prefer_node[self.all_ms_list[ms_priority[idx]]] = cur_node
                else:
                    ms_prefer_node[self.all_ms_list[ms_priority[idx]]] = edge_node[len(edge_node)-idx%self.NODE_NUM-1]
        return ms_prefer_node
    def get_user_first_arrival_node(self):
        user_first_node_list = []
        for user in self.user_list:
            user_first_node_list.append(self.ms_first_node[user.request_chain[0]])
        return user_first_node_list
    def get_ms_global_arrival_rates(self):
        ms_global_arrival_rates = np.zeros(self.MS_AIMS_NUM)
        for user in self.user_list:
            for ms_item in user.request_chain:
                if isinstance(ms_item,MS):
                    ms_global_arrival_rates[ms_item.id] += user.lamda
                else:
                    ms_global_arrival_rates[ms_item.id+self.MS_NUM] += user.lamda
        return ms_global_arrival_rates
    def get_Ms_types_instances(self):
        ms_instances = []
        aims_instances = []
        for ms_item in self.all_ms_list:
            if isinstance(ms_item,MS):
                ms_id = ms_item.id
            else:
                ms_id = ms_item.id+self.MS_NUM
            self.Ms_types_demand[ms_id] = math.ceil((self.Ms_types_global_arrival_rates[ms_id]+1)/ms_item.alpha)
        for aims_i in range(self.MS_NUM, self.MS_AIMS_NUM):
            for ms_instances_num in range(self.Ms_types_demand[aims_i]):
                aims_instances.append(self.all_ms_list[aims_i])
        for ms_i in range(self.MS_NUM):
            for ms_instances_num in range(self.Ms_types_demand[ms_i]):
                ms_instances.append(self.all_ms_list[ms_i])
        # random.seed(1)
        # random.shuffle(aims_instances)
        # random.shuffle(ms_instances)
        all_ms_instances = [item for sublist in [aims_instances, ms_instances] for item in sublist]
        return all_ms_instances

    def init_deployment_and_resource_solution(self):
        for ms_item in self.all_ms_list:
            if isinstance(ms_item, MS):
                self.MS_instances_deployed_on_server[ms_item.id][self.ms_first_node[ms_item]] += 1
            else:
                self.MS_instances_deployed_on_server[ms_item.id + self.MS_NUM][self.ms_first_node[ms_item]] += 1
            self.CPU_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_cpu()  # 初始化剩余cpu资源
            self.GPU_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_gpu()  # 初始化剩余gpu资源
            self.Memory_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_memory() # 初始化剩余memory资源

    def greed_solution(self):
        for ms_item in self.all_ms_list:
            if isinstance(ms_item, MS):
                self.MS_instances_deployed_on_server[ms_item.id][self.ms_first_node[ms_item]] += self.Ms_types_demand[ms_item.id]
                self.CPU_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_cpu()*self.Ms_types_demand[ms_item.id]  # 初始化剩余cpu资源
                self.GPU_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_gpu() *self.Ms_types_demand[ms_item.id] # 初始化剩余gpu资源
                self.Memory_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_memory()*self.Ms_types_demand[ms_item.id]  # 初始化剩余memory资源
            else:
                self.MS_instances_deployed_on_server[ms_item.id + self.MS_NUM][self.ms_first_node[ms_item]] += self.Ms_types_demand[ms_item.id+ self.MS_NUM]
                self.CPU_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_cpu() * self.Ms_types_demand[ms_item.id+ self.MS_NUM]  # 初始化剩余cpu资源
                self.GPU_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_gpu() * self.Ms_types_demand[ms_item.id+ self.MS_NUM]  # 初始化剩余gpu资源
                self.Memory_resource_on_service[self.ms_first_node[ms_item]] -= ms_item.get_memory() * self.Ms_types_demand[ms_item.id+ self.MS_NUM]  # 初始化剩余memory资源
    def get_deployment_graph(self):
        """
        提取节点图信息
        :return:
        """
        # 获取边的信息,服务器之间的依赖，可达性
        target_nodes = []
        source_nodes = []
        for i in range(self.NODE_NUM):
            for j in range(self.NODE_NUM):
                if self.graph[i][j] == 1:
                    target_nodes.append(i)
                    source_nodes.append(j)
        edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long).to(self.device)
        # 获取节点信息
        node_features = []
        deploy_node = np.array(self.MS_instances_deployed_on_server).T
        for node_idx in range(self.NODE_NUM):
            node_info = []
            # 微服务部署信息
            for item in deploy_node[node_idx]:
                node_info.append(item)
            node_features.append(node_info)
        return Data(x=torch.tensor(node_features, dtype=torch.float32).to(self.device), edge_index=edge_index)
    def get_rout_graph(self, done):
        """
            获取当前微服务的转发情况图,节点特征表示该服务器接受该微服务的概率
            :return:
            """
        current_ms = self.Ms_types_instances[self.current_ms_index]
        service_rout_Datalist = []
        for user in self.user_list:
            node_features = [[0.0 for _ in range(self.MS_AIMS_NUM)] for _ in range(self.NODE_NUM)]  # 所有出现过的点标记为1否则为0
            x = []
            y = []
            this_service_rout = self.Service_rout_table.get(user) # user的路由方案
            # this_service_rout = self.get_each_request_rout(deploy).get(user)  # user的路由方案
            last_ms_rout_node_p = {}
            for current_ms_rout in this_service_rout:
                new_ms_rout_node_p = {}
                for item in current_ms_rout:
                    pre_node = item[0]
                    cur_node = item[1]
                    cur_ms = item[2]
                    p = item[3]
                    if this_service_rout.index(current_ms_rout) > 0:
                        node_features[cur_node][cur_ms] += p * last_ms_rout_node_p[pre_node]
                        if cur_node not in new_ms_rout_node_p:
                            new_ms_rout_node_p[cur_node] = p * last_ms_rout_node_p[pre_node]
                        else:
                            new_ms_rout_node_p[cur_node] += p * last_ms_rout_node_p[pre_node]
                    else:
                        node_features[cur_node][cur_ms] += p
                        new_ms_rout_node_p[cur_node] = p
                    # 特殊情况处理成自环
                    if pre_node != -1:
                        x.append(pre_node)
                        y.append(cur_node)
                    if item==current_ms_rout[-1]:
                        last_ms_rout_node_p = new_ms_rout_node_p.copy()
            edge_index = torch.tensor([x, y], dtype=torch.long).to(self.device)  # 边
            node_features = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            if not done:
                if current_ms in user.request_chain:
                    service_rout_Datalist.append(Data(x=node_features, edge_index=edge_index))
            else:
                service_rout_Datalist.append(Data(x=node_features, edge_index=edge_index))
        # service_rout_Data_set = DataLoader(service_rout_Datalist, batch_size=self.USER_NUM, shuffle=True)
        service_rout_Data_set = Batch.from_data_list(service_rout_Datalist)
        return service_rout_Data_set
    def get_ms_graph(self):
        """
        创建当前请求的服务依赖图
        :return:
        """
        # 生成微服务实例依赖
        dependency = np.zeros((self.MS_AIMS_NUM, self.MS_AIMS_NUM), dtype=int)
        for i in range(self.MS_AIMS_NUM):
            dependency[i][i] = 1
        for user in self.user_list:
            request = user.request_chain
            pre_ms = request[0].id
            for ms_item in request[1:]:
                if isinstance(ms_item, MS):
                    dependency[pre_ms][ms_item.id] = 1
                    dependency[ms_item.id][pre_ms] = 1
                    pre_ms = ms_item.id
                else:
                    dependency[pre_ms][ms_item.id + self.MS_NUM] = 1
                    dependency[ms_item.id + self.MS_NUM][pre_ms] = 1
                    pre_ms = ms_item.id + self.MS_NUM
        # 生成节点信息
        x = []
        for ms in self.all_ms_list:
            this_ms = []
            this_ms.append(ms.id if isinstance(ms, MS) else ms.id + self.MS_NUM)
            this_ms.append(ms.get_cpu())
            this_ms.append(ms.get_gpu())
            this_ms.append(ms.get_memory())
            this_ms.append(ms.data)
            this_ms.append(ms.get_alpha())
            x.append(this_ms)
        # 生成边信息,微服务依赖关系
        source_nodes = []
        target_nodes = []
        for org_ms_id in range(self.MS_AIMS_NUM):
            for tar_ms_idx in range(self.MS_AIMS_NUM):
                if dependency[org_ms_id][tar_ms_idx] != 0:
                    source_nodes.append(org_ms_id)
                    target_nodes.append(tar_ms_idx)
        edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long).to(self.device)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return Data(x=x, edge_index=edge_index)

    def GNN_state(self,done):
        '''
        ms_deployment_on_node：部署方案和服务器拓扑结构
        service_rout：当前服务的路由拓扑结构
        ms_dependency：微服务调用图
        :param state:
        :return:
        '''
        ms_deployment_on_node = self.get_deployment_graph()
        service_rout = self.get_rout_graph(done)
        return ms_deployment_on_node, service_rout, self.GNN_ms_dependency
    def get_mark(self, ms):
        mark = []
        ms_cpu = ms.get_cpu()
        ms_gpu = ms.get_gpu()
        ms_mem = ms.get_memory()
        for node in self.node_list:
            if ms_cpu <= self.CPU_resource_on_service[node.id] \
                    and ms_gpu <= self.GPU_resource_on_service[node.id] \
                    and ms_mem < self.Memory_resource_on_service[node.id]:
                mark.append(1)
            else:
                mark.append(0)
        return mark
    def MLP_state(self,done):
        '''
        向量特征:当前部署的微服务信息，服务信息--->state[:7+MAX_LENGTH]
        :param state:
        :return:
        '''
        if not done:
            next_ms = self.Ms_types_instances[self.current_ms_index+1]
            if isinstance(next_ms, MS):
                next_ms_id = next_ms.id
            else:
                next_ms_id = next_ms.id + self.MS_NUM
            current_ms_info = [next_ms_id,
                               next_ms.get_cpu(),
                               next_ms.get_gpu(),
                               next_ms.get_memory(),
                               next_ms.data,
                               next_ms.alpha]
            node_with_can_deploy_current_ms = self.get_mark(next_ms)
            traffic = self.get_microservices_traffic_distribution()[next_ms_id].tolist()
        else:
            current_ms_info = [-1,-1,-1,-1,-1,-1]
            traffic = [0,0,0,0,0,0,0,0,0,0]
            node_with_can_deploy_current_ms = [0,0,0,0,0,0,0,0,0,0]
        return traffic+node_with_can_deploy_current_ms

    def get_state(self,done):
        GNN_deployment, GNN_rout, GNN_ms_dependency = self.GNN_state(done)
        MLP_info = self.MLP_state(done)
        return [MLP_info, GNN_deployment, GNN_rout, GNN_ms_dependency]
    def reset(self):
        self.MS_instances_deployed_on_server = [[0 for _ in range(self.NODE_NUM)] for _ in range(self.MS_AIMS_NUM)]  # 网络中的部署方案
        self.CPU_resource_on_service = [node.get_cpu() for node in self.node_list]
        self.GPU_resource_on_service = [node.get_gpu() for node in self.node_list]
        self.Memory_resource_on_service = [node.get_memory() for node in self.node_list]
        self.init_deployment_and_resource_solution()  # 初始化部署方案
        self.Service_rout_table = self.get_each_request_rout()  # 网络中服务的路由方案
        self.last_step_sum_delay = self.get_total_delay()
        self.current_ms_index = 0  # 当前正在部署的微服务实例在实例集合中的位置
        state = self.get_state(False)
        return state
    # 更新部署和路由方案以及资源剩余情况
    def update_deployment_resource_and_service_rout(self, action, ms):
        if isinstance(ms, MS):
            ms_id = ms.id
        else:
            ms_id = ms.id + self.MS_NUM
        self.MS_instances_deployed_on_server[ms_id][action] += 1
        self.CPU_resource_on_service[action] -= ms.get_cpu()
        self.GPU_resource_on_service[action] -= ms.get_gpu()
        self.Memory_resource_on_service[action] -= ms.get_memory()
        self.Service_rout_table = self.get_each_request_rout()
    def get_done(self):
        if self.current_ms_index!=self.Ms_instance_sum-1:
            done = False
        else:
            done = True
        return done
    def get_reward(self,done):
        if not done:
            this_step_sum_delay = self.get_total_delay()
            if self.last_step_sum_delay==this_step_sum_delay:
                reward = 10
            else:
                reward = (self.last_step_sum_delay-this_step_sum_delay)*10
            self.last_step_sum_delay = this_step_sum_delay
        else:
            this_episode_sum_delay = self.get_total_delay()
            reward = (self.last_step_sum_delay-this_episode_sum_delay)
            reward += (self.last_episode_sum_delay - this_episode_sum_delay) + (self.min_episode_sum_delay-this_episode_sum_delay)
            self.last_episode_sum_delay = this_episode_sum_delay
            self.min_episode_sum_delay = min(self.last_episode_sum_delay, this_episode_sum_delay)
        return reward
    def agent_step(self,action):
        self.update_deployment_resource_and_service_rout(action,self.Ms_types_instances[self.current_ms_index])
        done = self.get_done()
        next_state = self.get_state(done)  # [MLP,GNN_D,GNN_R,GNN_M]
        reward = self.get_reward(done)
        self.current_ms_index += 1
        return next_state,reward,done

    # def Highest_return_trajectory(self):



    # 下面是一些重要计算
    # 优化路由节点
    def optimize_rout_node(self, ms_node_dict, request):
        '''
        去除不可达的节点
        :param ms_node_dict:
        :param request:
        :param node_graph:
        :return:
        '''
        result = ms_node_dict
        first_ms = request[0]
        list_node = []  # 后续进行递归检查需要用到，每一行保存这相应微服务的部署服务器

        for ms_item in request[1:]:
            node1 = ms_node_dict.get(first_ms)
            list_node.append(node1)
            node2 = ms_node_dict.get(ms_item)
            lag = {i: 0 for i in node2}
            if len(node1) == 0:
                new_node = []
                list_node.append(new_node)
                result[ms_item] = new_node

            else:
                new_node = node2.copy()
                for i in node1:
                    for j in node2:
                        # print("服务器id",i.id,j.id)
                        if self.graph[i.id][j.id] != 0:
                            # print("yes")
                            lag[j] += 1
                # print(lag)
                for item in lag:
                    if lag.get(item) == 0:
                        new_node.remove(item)
                # print(new_node==node2)
                list_node.append(new_node)
                result[ms_item] = new_node
            first_ms = ms_item
        sur_ms = request[-1]
        for ms_item in request[::-1]:
            if ms_item == sur_ms:
                continue
            node1 = ms_node_dict.get(sur_ms)
            list_node.append(node1)
            node2 = ms_node_dict.get(ms_item)
            lag = {i: 0 for i in node2}
            new_node = node2.copy()
            if len(node1) != 0:
                for i in node1:
                    for j in node2:
                        # print("服务器id",i.id,j.id)
                        if self.graph[i.id][j.id] != 0:
                            # print("yes")
                            lag[j] += 1
                # print(lag)
                for item in lag:
                    if lag.get(item) == 0:
                        new_node.remove(item)
                # print(new_node==node2)
                list_node.append(new_node)
                result[ms_item] = new_node
            sur_ms = ms_item
        return result
    # 计算转发概率
    def cal_probability(self, node2, ms, ms_node_list):
        '''
        计算node1转发到node2上的转发概率
        :param node2:
        :param ms:
        :param node_list: 部署了ms的服务器集合
        :param node_bandwidth:
        :param deploy: 服务部署方案
        :return:
        '''
        total_ma_image = 0
        total_bandwidth = 0
        if isinstance(ms, MS):
            ms_id = ms.id
        else:
            ms_id = ms.id + self.MS_NUM
        for item in ms_node_list:
            total_bandwidth += item.bandwidth
            total_ma_image += self.MS_instances_deployed_on_server[ms_id][item.id]
        # p = (node2.bandwidth + deploy[ms.id][node2.id]) / (total_bandwidth + total_ma_image)
        p = round((self.MS_instances_deployed_on_server[ms_id][node2.id])
                  / (total_ma_image), 3)
        return ms, node2, p
    # 获得完整的路由方案
    def get_each_request_rout(self):
        '''
        根据部署方案和服务请求生成每一条服务请求的处理路径图，每一个节点表示服务器，每一个边表示转发概率
        第一个节点是固定的，通过get_first_node（）函数获得
        :param deploy:
        :param users:
        :param requests:
        :return: 返回服务请求路由路径集合，每一条服务请求的路由路径图用邻接表存储。
        邻接表中用元组来表示路由转发(上一个节点,当前节点, 当前需要处理的微服务,接收概率)
        '''
        all_user_rout = {}
        for user in self.user_list:
            request = user.request_chain
            idx = 0
            # node_list[node_idx] = EDGE_NODE(first_node)
            # 生成当前服务请求中各个微服务所在的节点集合
            # ms_node_dict: ms1:[node1, node2],ms2:[...]
            ms_node_dict = {}
            ms_node_dict[request[0]] = [self.node_list[self.user_first_arrival_node_list[user.id]]]
            for ms_item in request[1:]:
                this_ms_node = []
                if isinstance(ms_item, MS):
                    current_node = self.MS_instances_deployed_on_server[ms_item.id]
                else:
                    current_node = self.MS_instances_deployed_on_server[ms_item.id + self.MS_NUM]
                for node in self.node_list:
                    if current_node[node.id] != 0:
                        this_ms_node.append(node)
                        # this_ms_node.append(EDGE_NODE(node_idx))
                ms_node_dict[ms_item] = this_ms_node
                idx += 1
            ms_node_dict = self.optimize_rout_node(ms_node_dict, request)
            all_node_list = []  # 存储了路由转发图中所有出现的节点，节点id会有重复
            for item in ms_node_dict:
                some_node = ms_node_dict.get(item)
                for node in some_node:
                    all_node_list.append(node)
            # print(all_node_list)
            this_user_rout_path_p = []
            # 第一个微服务的转发情况需要特殊处理
            pre_node_list = ms_node_dict.get(request[0]).copy()
            first_ms = request[0]
            if isinstance(first_ms,MS):
                first_ms_rout = [(-1, pre_node_list[0].id, request[0].id, 1.0)]
            else:
                first_ms_rout = [(-1, pre_node_list[0].id, request[0].id+self.MS_NUM, 1.0)]
            this_user_rout_path_p.append(first_ms_rout)
            for item in request[1:]:
                all_node_of_this_ms = ms_node_dict.get(item).copy()
                ms_rout = []
                for node in pre_node_list:
                    new_all_node_of_this_ms = []
                    for node1 in all_node_of_this_ms:
                        if self.graph[node.id][node1.id] != 0:
                            new_all_node_of_this_ms.append(node1)
                    for node2 in new_all_node_of_this_ms:
                        _, _, p = self.cal_probability(node2, item, new_all_node_of_this_ms)
                        if isinstance(item,MS):
                            ms_id = item.id
                        else:
                            ms_id = item.id+self.MS_NUM
                        rout = (node.id, node2.id, ms_id, p)
                        ms_rout.append(rout)
                    if not ms_rout:
                        continue
                pre_node_list = all_node_of_this_ms
                this_user_rout_path_p.append(ms_rout)
            all_user_rout[user] = this_user_rout_path_p
        return all_user_rout
    def jiechen(self, n):
        k = 1
        if n == 0:
            return 1
        else:
            for i in range(1, n + 1):
                k *= i
            return k
    def get_microservices_traffic_distribution(self):
        M_T_D = np.zeros((self.MS_AIMS_NUM, self.NODE_NUM))
        for user in self.user_list:
            this_request_rout = self.Service_rout_table.get(user)
            ori_traffic = user.lamda
            request = user.request_chain
            traffic = {}
            for idx in range(len(this_request_rout)):
                new_traffic = {}
                ms = request[idx]
                if isinstance(ms, MS):
                    ms_id = ms.id
                else:
                    ms_id = ms.id + self.MS_NUM
                for (last_node, this_node, _, p) in this_request_rout[idx]:
                    if last_node == -1:
                        M_T_D[ms_id][this_node] += ori_traffic * p
                        if this_node in traffic:
                            traffic[this_node] += ori_traffic * p
                        else:
                            traffic[this_node] = ori_traffic * p
                    else:
                        M_T_D[ms_id][this_node] += traffic[last_node] * p
                        if this_node in new_traffic:
                            new_traffic[this_node] += traffic[last_node] * p
                        else:
                            new_traffic[this_node] = traffic[last_node] * p
                if idx > 0:
                    traffic = new_traffic.copy()
        return M_T_D
    def get_total_acc_delay(self):
        acc_delay = np.zeros(self.USER_NUM)
        for user in self.user_list:
            ms_data = user.request_chain[0].data
            B = self.node_list[self.user_first_arrival_node_list[user.id]].bandwidth
            acc_delay[user.id] = ms_data / B * user.lamda
        return sum(acc_delay), acc_delay
    def get_total_sojourn_delay(self, traffic):
        ms_sojourn_time = np.zeros((self.MS_AIMS_NUM, self.NODE_NUM))
        for ms_idx in range(self.MS_AIMS_NUM):
            for node_idx in range(self.NODE_NUM):
                if traffic[ms_idx][node_idx] == 0:
                    continue
                num = int(self.MS_instances_deployed_on_server[ms_idx][node_idx])
                if num == 0:
                    continue
                ms = self.all_ms_list[ms_idx]
                alpha = ms.alpha
                lamda = traffic[ms_idx][node_idx]
                rh0 = lamda / alpha
                rh1 = lamda / (num * alpha)
                if rh1 > 0.0 and rh1 < 0.99:
                    v1 = 0
                    for n in range(num):
                        v2 = self.jiechen(n)
                        v3 = math.pow(rh0, n) / v2
                        v1 += v3
                    v5 = self.jiechen(num)
                    p0 = math.pow((v1 + math.pow(rh0, num) / (v5 * (1 - rh1))), -1)
                    ms_proc_delay = 1 / alpha + rh1 * math.pow(rh0, num) * p0 / (lamda * v5 * math.pow((1 - rh1), 2))
                else:
                    if isinstance(ms, MS):
                        ms_proc_delay = 5
                    else:
                        ms_proc_delay = 10
                ms_sojourn_time[ms_idx][node_idx] = ms_proc_delay
        return sum([sum(row) for row in ms_sojourn_time]), ms_sojourn_time
    def get_total_communication_delay(self):
        total_communication_delay = np.zeros(self.USER_NUM)
        for user in self.user_list:
            request = user.request_chain
            rout = self.Service_rout_table.get(user)
            lamda_of_node = {self.user_first_arrival_node_list[user.id]:user.lamda}
            for idx in range(1,len(rout)):
                if not rout[idx]:
                    if isinstance(request[idx], MS):
                        total_communication_delay[user.id] += 5
                    else:
                        total_communication_delay[user.id] += 5
                else:
                    new_lamda_of_node = {}
                    for ele2 in rout[idx]:
                        pre_node = self.node_list[ele2[0]]
                        node = self.node_list[ele2[1]]
                        ms = request[idx-1]
                        p = ele2[3]
                        ms_lamda = lamda_of_node[pre_node.id] * p
                        if node.id in new_lamda_of_node:
                            new_lamda_of_node[node.id] += ms_lamda
                        else:
                            new_lamda_of_node[node.id] = ms_lamda
                        # 计算通信延迟
                        if pre_node.id!=node.id:
                            total_communication_delay[user.id] += ms.data / node.bandwidth * ms_lamda
                    lamda_of_node = new_lamda_of_node
        return sum(total_communication_delay), total_communication_delay
    def get_total_return_delay(self):
        total_return_delay = np.zeros(self.USER_NUM)
        for user in self.user_list:
            lamda = user.lamda
            rout = self.Service_rout_table.get(user)
            lamda_of_node = {}
            for idx in range(len(rout)):
                if idx == 0:
                    for ele1 in rout[idx]:
                        node = self.node_list[ele1[1]]
                        p = ele1[3]
                        ms_lamda = lamda * p
                        # 保存本次流量分流情况
                        lamda_of_node[node.id] = ms_lamda
                else:
                    new_lamda_of_node = {}
                    for ele2 in rout[idx]:
                        pre_node = self.node_list[ele2[0]]
                        node = self.node_list[ele2[1]]
                        p = ele2[3]
                        ms_lamda = lamda_of_node[pre_node.id] * p
                        if node.id in new_lamda_of_node:
                            new_lamda_of_node[node.id] += ms_lamda
                        else:
                            new_lamda_of_node[node.id] = ms_lamda
                    lamda_of_node = new_lamda_of_node
            # 服务请求的发送时延需要单独计算(传输与传送)
            for key, value in lamda_of_node.items():
                node = self.node_list[key]
                ms = user.request_chain[-1]
                rec_delay = ms.data / node.bandwidth * value
                total_return_delay[user.id] += rec_delay
        return sum(total_return_delay), total_return_delay
    # 计算服务总时延
    def get_total_delay(self):
        '''
        网络时延的组成：发送时延和接收时延，等待时延，传输时延，传送时延
        :param users: 用户集
        :param node_list: 节点集
        :param all_ms_proc_delay: 微服务在服务器上的等待延迟
        :param all_user_rout: 请求路由表
        :param bandwidth: 带宽
        :param ms_data: 数据大小
        :return:
        '''
        ms_traffic = self.get_microservices_traffic_distribution()
        acc_delay,_ = self.get_total_acc_delay()
        sojourn_delay,_ = self.get_total_sojourn_delay(ms_traffic)
        communication_delay,_ = self.get_total_communication_delay()
        return_delay,_ = self.get_total_return_delay()
        return acc_delay+sojourn_delay+communication_delay+return_delay
    # 计算网络中服务器负载情况
    def cal_load_balance(self):
        u_cpu = (np.array([node.get_cpu() for node in self.node_list])-np.array(self.CPU_resource_on_service)) \
                / np.array([node.get_cpu() for node in self.node_list])
        u_gpu = np.zeros((self.NODE_NUM))
        for node in self.node_list:
            if node.get_gpu() != 0:
                u_gpu[node.id] = (node.get_gpu()-self.GPU_resource_on_service[node.id]) / node.get_gpu()
        u_mem = (np.array([node.get_memory() for node in self.node_list])-np.array(self.Memory_resource_on_service)) \
                / np.array([node.get_memory() for node in self.node_list])
        avg_cpu = np.full(u_cpu.size, np.mean(u_cpu))
        avg_gpu = np.full(u_gpu.size, np.mean(u_gpu))
        avg_mem = np.full(u_mem.size, np.mean(u_mem))
        # print(u_cpu+u_gpu+u_mem)
        v_cpu = np.mean((u_cpu - avg_cpu) ** 2)
        v_gpu = np.mean((u_gpu - avg_gpu) ** 2)
        v_mem = np.mean((u_mem - avg_mem) ** 2)
        return v_cpu + v_gpu + v_mem

    def greedy_deployment(self):
        current_ms = self.Ms_types_instances[self.current_ms_index]
        return self.ms_first_node[current_ms]



