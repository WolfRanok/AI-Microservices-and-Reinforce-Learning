import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from ENV import *
from Object_Parameter_DEF import MA_AIMS_NUM, NODE_NUM

from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.utils import remove_self_loops, add_self_loops


NODE_FEATURES_NUM = 7 + MA_AIMS_NUM # 服务器图的节点特征数量
DEPENDENCY_FEATURES_NUM = 7  # 用户请求依赖图的节点特征数量
ROUT_FEATURES_NUM = 1  # 路由概率图的节点特征数量
Extra_FEATURES_NUM = 9  # 表示用户以及请求链的向量信息维度

# 动态图神经网络子模型（用于节点图、路由图、微服务依赖关系图的生成）
# 检测 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_channels, 32)
        self.pool1 = TopKPooling(32, ratio=0.8)  # 添加TopKPooling层，保留80%的节点
        self.conv2 = SAGEConv(32, out_channels)

    def forward(self, data):
        # 将数据移动到与模型相同的设备上
        data.x = data.x.to(device, dtype=torch.float32)

        data.edge_index = data.edge_index.to(device)

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, None)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv2(x, edge_index)
        x = torch.mean(x, dim=0, keepdim=True)
        return x

# 该网络用于处理用户的向量信息
class ExtraInfoNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExtraInfoNet, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # 将数据移动到与模型相同的设备上
        x = x.to(device, dtype=torch.float32)
        x = self.fc1(x)
        x = F.relu(x)
        return x

class LSTM_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lstm_hidden=128, fc_hidden=64):
        super(LSTM_Actor, self).__init__()
        # 这里需要构建三个图神经网络模型以及一个普通神经网络模型的图
        self.node_network = GCN(NODE_FEATURES_NUM, 32)
        self.dependency_network = GCN(DEPENDENCY_FEATURES_NUM, 8)
        self.rout_network = GCN(ROUT_FEATURES_NUM, 16)
        self.extra_network = ExtraInfoNet(Extra_FEATURES_NUM, 8)

        # 还有一个actor层
        self.fc_action = nn.Linear(32 + 8 + 16 + 8, action_dim)

        # 将模型移动到 GPU 上
        self.to(device)

    def forward(self, state, hidden=None):
        # 批量化处理
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        q_values = []

        for s in state:
            # 处理图像数据
            graph_data = State2GraphData(s)

            # 获取四个子模型的输出数据
            out_node_features = self.node_network(graph_data.graph_node)
            out_dependency_data = self.dependency_network(graph_data.graph_dependency)
            out_rout_data = self.rout_network(graph_data.ms_rout)
            out_extra_data = self.extra_network(graph_data.ms_request).unsqueeze(0) # 将 out_extra_data 从一维张量转换为二维张量

            combined_output = torch.cat([out_node_features, out_dependency_data, out_rout_data, out_extra_data], dim=1)

            # 给每一个动作产生行动概率
            action_logits = self.fc_action(combined_output)
            action_probs = F.softmax(action_logits, dim=1)
            q_values.append(action_probs)
        return torch.cat(q_values, dim=0)

    def select_action(self, state):
        """
        对指定的服务类型按照概率选择一个行动
        :param index: 服务类型
        :param action_probabilities: 行动概率分布
        :return: action
        """
        with torch.no_grad():
            action_probabilities = self.forward(state)
            action = torch.multinomial(action_probabilities[0], num_samples=1).item()
        return action


class LSTM_Critic(nn.Module):
    def __init__(self, state_dim, lstm_hidden=128, fc_hidden=64):
        super(LSTM_Critic, self).__init__()
        # 这里需要构建三个图神经网络模型以及一个普通神经网络模型的图
        self.node_network = GCN(NODE_FEATURES_NUM, 32)
        self.dependency_network = GCN(DEPENDENCY_FEATURES_NUM, 8)
        self.rout_network = GCN(ROUT_FEATURES_NUM, 16)
        self.extra_network = ExtraInfoNet(Extra_FEATURES_NUM, 8)

        # critic评价层
        self.fc_critic = nn.Linear(32 + 8 + 16 + 8, 1)

        # 将模型移动到 GPU 上
        self.to(device)

    def forward(self, state, hidden=None):
        # 将数据变为二维，从而支持批量处理
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        q_values = []

        # print("state.shape: ", state.shape)
        for s in state:
            # 处理图像数据
            graph_data = State2GraphData(s)

            # 获取四个子模型的输出数据
            out_node_features = self.node_network(graph_data.graph_node)
            out_dependency_data = self.dependency_network(graph_data.graph_dependency)
            out_rout_data = self.rout_network(graph_data.ms_rout)
            out_extra_data = self.extra_network(graph_data.ms_request).unsqueeze(0) # 将 out_extra_data 从一维张量转换为二维张量

            combined_output = torch.cat([out_node_features, out_dependency_data, out_rout_data, out_extra_data], dim=1)

            # 评估Q值
            q_value = self.fc_critic(combined_output)
            q_values.append(q_value)
        return torch.cat(q_values, dim=0)

# class LSTM_Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, lstm_hidden=128, fc_hidden=64):
#         super(LSTM_Actor, self).__init__()
#         # LSTM层处理时序状态
#         self.lstm = nn.LSTM(state_dim, lstm_hidden, batch_first=True)
#         # 全连接层生成动作概率
#         self.fc = nn.Sequential(
#             nn.Linear(lstm_hidden, fc_hidden),
#             nn.ReLU(),
#             nn.Linear(fc_hidden, action_dim),
#         )
#
#     def forward(self, state, hidden=None):
#         # state: (batch_size, state_dim)
#         if len(state.shape) == 1:
#             state = state.unsqueeze(0)
#         # LSTM处理时序特征
#         lstm_out, hidden = self.lstm(state, hidden)
#         # 生成动作概率分布
#         action_probs = self.fc(lstm_out)
#         action_probs = F.softmax(action_probs, dim=-1)
#         return action_probs
#
#     def select_action(self, state):
#         """
#         对指定的服务类型按照概率选择一个行动
#         :param index: 服务类型
#         :param action_probabilities: 行动概率分布
#         :return: action
#         """
#         with torch.no_grad():
#             action_probabilities = self.forward(state)
#             action = torch.multinomial(action_probabilities[0], num_samples=1).item()
#         return action
#
#
# class LSTM_Critic(nn.Module):
#     def __init__(self, state_dim, lstm_hidden=128, fc_hidden=64):
#         super(LSTM_Critic, self).__init__()
#         # LSTM处理状态
#         self.lstm = nn.LSTM(state_dim, lstm_hidden, batch_first=True)
#         # 融合状态和动作特征
#         self.fc = nn.Sequential(
#             nn.Linear(lstm_hidden, fc_hidden),
#             nn.ReLU(),
#             nn.Linear(fc_hidden, 1)
#         )
#
#     def forward(self, state, hidden=None):
#         if len(state.shape) == 1:
#             state = state.unsqueeze(0)
#         # LSTM处理状态
#         lstm_out, hidden = self.lstm(state, hidden)
#         # 拼接状态特征和动作
#         # action = action.reshape(len(action), -1)
#         # combined = torch.cat([lstm_out, action], dim=1)
#         # 评估Q值
#         q_value = self.fc(lstm_out)
#         return q_value
class MLP_ACNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor分支（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)  # 离散动作
        )

        # Critic分支（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 状态价值V(s)
        )

    def forward(self, states):
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        features = self.shared(states)
        actions_probs = self.actor(features)
        values = self.critic(features)
        return actions_probs, values

    def select_action(self, state):
        """
        对指定的服务类型按照概率选择一个行动
        :param index: 服务类型
        :param action_probabilities: 行动概率分布
        :return: action
        """
        with torch.no_grad():
            action_probabilities, _ = self.forward(state)
            action = torch.multinomial(action_probabilities[0], num_samples=1).item()
        return action


class GNNActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim, num_heads=2):
        super(GNNActor, self).__init__()
        # GNN层：使用GAT（图注意力网络）提取节点特征
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)
        # 策略输出层
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 假设动作空间是离散的
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # GNN前向传播
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # 输出动作概率分布（每个节点的动作）
        return self.policy_head(x)


class GNNCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GNNCritic, self).__init__()
        # GNN层：使用GCN（图卷积网络）提取全局状态特征
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 价值输出层
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # GNN前向传播
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # 全局池化得到图级特征
        x = global_mean_pool(x, batch)
        # 输出状态价值
        return self.value_head(x)


class State2GraphData:
    """
    该类实现将数据转化为图神经网络可理解的数据
    以及其他的向量信息
    """
    start_deploy = 9 + MA_AIMS_NUM + MA_AIMS_NUM * NODE_NUM

    def __init__(self, state):

        self.state = state.to(torch.float32)

        self.deploy_state = self.state[self.start_deploy: self.start_deploy + MA_AIMS_NUM * NODE_NUM]  # 微服务部署情况
        self.graph_node = self.get_graph_node()  # 获取节点图信息
        self.graph_dependency = self.get_graph_dependency()  # 获取当前用户请求的依赖信息
        self.ms_request = self.get_ms_request()  # 获取其他的向量信息(有关微服务和请求链的信息)
        self.ms_rout = self.get_ms_rout()  # 获取当前微服务的转发情况

    def get_ms_rout(self):
        """
        获取当前微服务的转发情况图
        :return:
        """
        deploy = self.deploy_state.clone()
        deploy = deploy.reshape(MA_AIMS_NUM, NODE_NUM)
        current_user = users[0]
        current_request = requests.get(current_user)  # 服务请求
        current_request_rout = get_this_request_rout(deploy, current_user, current_request)
        current_ms = self.state[0]
        for rout in current_request_rout:
            if rout and rout[0][2] == current_ms:   # 找到存有当前微服务的转发表信息
                node_list = [False for _ in range(NODE_NUM)]    # 用于判断是否有出现当前的节点
                x = []
                y = []
                edge_weights = []
                for a, b, _, p in rout:
                    # 特殊情况处理成自环
                    if a == -1:
                        x.append(b)
                    else:
                        x.append(a)
                        node_list[a] = True

                    y.append(b)
                    node_list[b] = True
                    edge_weights.append(p)  # 将概率作为边权

                edge_index = torch.tensor([x, y], dtype=torch.long) # 边
                edge_weights = torch.tensor(edge_weights,dtype=torch.float)  # 边权

                node_features = [[1] if flag else [0] for flag in node_list]    # 所有出现过的点标记为1否则为0
                node_features = torch.tensor(node_features,dtype=torch.float)

                return Data(x=node_features, edge_index=edge_index, edge_weights=edge_weights)

        # 说明还没有微服务的转发的情况
        node_features = [[0] for _ in range(NODE_NUM)]  # 所有出现过的点标记为0

        # 边全部处理成自环
        x = [i for i in range(NODE_NUM)]
        y = [i for i in range(NODE_NUM)]
        edge_index = torch.tensor([x, y], dtype=torch.long)
        node_features = torch.tensor(node_features, dtype=torch.float)
        # print("路径生成出错！state信息为：",self.state," deploy：",deploy, " current_ms:", current_ms)
        return Data(x=node_features,edge_index=edge_index)

    def get_ms_request(self):
        """
        补充一些不需要使用图神经网络的9个数据
        具体包括：微服务的 id、数据量、cpu、gpu、内存需求、第一个用户的 id（请求链的id）、请求到达率、最大容忍延迟、服务请求的响应时延
        :return:
        """
        extra_data = self.state[:9]
        return extra_data

    def get_graph_dependency(self):
        """
        创建当前请求的服务依赖图
        :return:
        """
        self.current_request_info = self.state[5: 4 + MA_AIMS_NUM + MA_AIMS_NUM * NODE_NUM]  # 获取用户请求的信息：用户的 id、请求到达率、最大容忍延迟、服务请求的响应时延、待部署微服务镜像情况、路由概率表
        user = users[int(self.current_request_info[0])]

        x = []
        edge_index = []

        request = requests.get(user)
        for ms in request:  # 从请求链中得到微服务信息
            this_ms = []
            this_ms.append(user.id)
            this_ms.append(ms.id if isinstance(ms, MS) else ms.id + MS_NUM)
            this_ms.append(data[ms.id])
            this_ms.append(ms.get_cpu())
            this_ms.append(ms.get_gpu())
            this_ms.append(ms.get_memory())
            this_ms.append(ms.get_alpha())
            x.append(this_ms)

        index = [row[1] for row in x]
        encoded_index_column = LabelEncoder().fit_transform(index)
        target_nodes = encoded_index_column[1:]
        source_nodes = encoded_index_column[:-1]

        edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        # print("**:", x)
        return Data(x=x, edge_index=edge_index)

    def get_graph_node(self):
        """
        提取节点图信息
        :return:
        """
        self.resource = get_resource_for_new_state(self.state)  # 服务器资源情况信息

        # 获取边的信息
        target_nodes = []
        source_nodes = []

        for i in range(len(graph)):
            for j in range(len(graph[i])):
                if graph[i][j] == 1:
                    target_nodes.append(i)
                    source_nodes.append(j)

        edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)

        node_features = []

        deploy_node = get_deploy_for_new_state(self.state.cpu())  # 微服务在不同服务器的部署情况
        for node in node_list:
            node_info = []

            # 一些初始信息
            node_info.append(node.id)  # 节点id
            node_info.append(bandwidth[node.id])  # 节点带宽
            node_info.append(node.x)  # 节点坐标
            node_info.append(node.y)

            # 需要从状态中得到节点的资源剩余信息
            # print(self.state.shape)
            # print(self.resource[NODE_NUM: NODE_NUM * 2].shape, node.id)
            one_node_cpu = self.resource[NODE_NUM: NODE_NUM * 2][node.id]
            one_node_gpu = self.resource[NODE_NUM * 3: NODE_NUM * 4][node.id]
            one_node_memory = self.resource[NODE_NUM * 5:][node.id]

            # 节点当前的三种资源情况
            node_info.append(one_node_cpu)
            node_info.append(one_node_gpu)
            node_info.append(one_node_memory)

            # 微服务部署信息
            for ms in range(MA_AIMS_NUM):  # 遍历所有的微服务
                node_info.append(deploy_node[ms][node.id])

            node_features.append(node_info)

        return Data(x=torch.tensor(node_features), edge_index=edge_index)

    def show(self, data, name, id_index=None):
        """
        可视化图形信息
        :param data: Data的实例化
        :return:
        """
        # 创建一个空的无向图
        G = nx.DiGraph()

        print(data)
        mp = []
        for i in range(len(data.x)):
            if id_index is not None:
                id = int(data.x[i][id_index].item())
            else:
                id = i
            mp.append(id)
            G.add_node(id)

        # 根据关联矩阵添加边
        for x,y in zip(data.edge_index[0], data.edge_index[1]):
            x = x.item()
            y = y.item()
            G.add_edge(mp[x], mp[y])
        # 绘制图
        pos = nx.spring_layout(G)  # 节点布局算法
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_labels(G, pos)

        plt.title(name)
        # 显示图形
        plt.axis('off')
        plt.show()


state_test = [ 1.00000000e+01,  2.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.50000000e+01,  9.00000000e+00,  1.20000000e+01,  1.00000000e+01,
        4.33393338e+00,  1.30000000e+01,  6.00000000e+00,  1.20000000e+01,
        1.00000000e+00,  7.00000000e+00,  1.00000000e+01, -1.00000000e+00,
       -1.00000000e+00, -1.00000000e+00, -1.00000000e+00, -1.00000000e+00,
       -1.00000000e+00, -1.00000000e+00, -1.00000000e+00, -1.00000000e+00,
       -1.00000000e+00, -1.00000000e+00, -1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.93939394e-01,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        6.06060606e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  4.19354839e-01,  0.00000000e+00,  3.54838710e-01,
        0.00000000e+00,  6.66666667e-01,  0.00000000e+00,  0.00000000e+00,
        2.25806452e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        4.20000000e-01,  0.00000000e+00,  2.20000000e-01,  0.00000000e+00,
        3.60000000e-01,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.87671233e-01,  5.60000000e-01,  5.00000000e-01,  4.40000000e-01,
        2.38095238e-01,  0.00000000e+00,  0.00000000e+00,  2.87671233e-01,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.10919071e-01,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        5.52509021e-01,  0.00000000e+00,  2.36571908e-01,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        2.00000000e+00,  0.00000000e+00,  2.00000000e+00,  0.00000000e+00,
        3.00000000e+00,  2.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  3.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  2.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  3.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        4.00000000e+00,  1.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  3.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  2.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  2.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  2.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  2.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  3.00000000e+00,  4.00000000e+00,
        4.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  2.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.00000000e+00,
        1.30000000e+01,  1.90000000e+01,  2.10000000e+01,  4.00000000e+00,
        2.60000000e+01,  1.90000000e+01,  2.00000000e+01,  2.00000000e+01,
        1.20000000e+01,  1.70000000e+01,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.30000000e+01,  2.40000000e+01,  2.90000000e+01,
        4.00000000e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.00000000e+00,  3.00000000e+00,  2.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.40000000e+01,  1.40000000e+01,
        1.40000000e+01,  0.00000000e+00,  0.00000000e+00,  2.40000000e+01,
        1.04000000e+02,  1.47000000e+02,  1.68000000e+02,  3.90000000e+01,
        2.08000000e+02,  1.82000000e+02,  1.49000000e+02,  1.38000000e+02,
        8.40000000e+01,  4.26000000e+02,  3.88000000e+02,  3.02000000e+02,
        3.31000000e+02,  3.63000000e+02,  6.72000000e+02,  6.63000000e+02,
        7.19000000e+02,  3.54000000e+02,  3.97000000e+02,  0.00000000e+00,
        1.40000000e+01,  1.00000000e+00,  2.00000000e+01,  2.00000000e+00,
        1.20000000e+01,  3.00000000e+00,  2.00000000e+01,  4.00000000e+00,
        1.00000000e+01,  5.00000000e+00,  9.00000000e+00,  6.00000000e+00,
        1.90000000e+01,  7.00000000e+00,  1.70000000e+01,  8.00000000e+00,
        1.90000000e+01,  9.00000000e+00,  6.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00]

if __name__ == '__main__':
    # state_dim = 24
    # action_dim = 10
    #
    # actor = LSTM_Actor(state_dim, action_dim)
    # critic = LSTM_Critic(state_dim, action_dim)
    #
    # # 输入状态 (batch_size=64, state_dim=24)
    # state = torch.randn(64, state_dim)
    # # Actor生成动作分布
    # probabilities = actor(state)
    # print(probabilities)
    # # 选择动作（这里使用随机采样作为示例）
    # actions = torch.multinomial(probabilities, num_samples=1).squeeze()
    # print(actions)
    #
    # # Critic评估状态-动作组合的价值
    # value = critic(state, probabilities)  # 注意添加动作维度
    # print(value)
    # print(value.shape)  # 输出：torch.Size([64, 1])

    # 图像信息可视化
    state_test = torch.tensor(state_test,dtype=torch.float)
    a = State2GraphData(state_test)
    a.show(a.graph_node, "graph_node",0)
    a.show(a.graph_dependency, "graph_dependency",1)
    a.show(a.ms_rout, "ms_rout")