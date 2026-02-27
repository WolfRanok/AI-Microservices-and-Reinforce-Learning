import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Dataset, DataLoader

from Environment import *

import torch_geometric.nn as geom_nn


# 动态图神经网络子模型（用于节点图、路由图、微服务依赖关系图的生成）
# 检测 GPU 是否可用

def init_weights(module):
    if isinstance(module, nn.Linear):
        # 对前两层使用 Xavier 初始化
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        # 偏置初始化为零
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # 最后一层（输出层）缩小权重范围
    if isinstance(module, nn.Linear) and module.out_features == 10:
        nn.init.xavier_uniform_(module.weight, gain=0.1)  # 缩小输出层初始化范围
        # 偏置初始化为零
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def init_gat_layer(gat_layer):
    # 初始化线性层
    nn.init.xavier_uniform_(gat_layer.lin_r.weight)
    if gat_layer.lin_r.bias is not None:
        nn.init.zeros_(gat_layer.lin_r.bias)
    if hasattr(gat_layer, 'lin_dst') and gat_layer.lin_dst is not None:
        nn.init.xavier_uniform_(gat_layer.lin_dst.weight)
    # 初始化注意力参数
    nn.init.normal_(gat_layer.att, std=0.1)

def init_lstm_layer(lstm_layer):
    for name, param in lstm_layer.named_parameters():
        if 'weight' in name:
            # 权重使用正交初始化
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # 偏置初始化为零，并设置遗忘门偏置为 1
            nn.init.zeros_(param)
            hidden_size = lstm_layer.hidden_size
            # 遗忘门偏置位置：[hidden_size : 2*hidden_size]
            param.data[hidden_size: 2 * hidden_size] = 1.0

class Deployment_Feature_Extractor(nn.Module):
    def __init__(self, env, args, hidden = 256, out_channels=8):
        super().__init__()
        self.conv = geom_nn.GATv2Conv(env.MS_AIMS_NUM, hidden,heads=2).to(env.device)
        self.conv2 = geom_nn.GATv2Conv(2*hidden, out_channels, heads=1).to(env.device)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        if args.use_orthogonal_init:
            init_gat_layer(self.conv)
            init_gat_layer(self.conv2)
        self.global_pool = global_mean_pool
    def forward(self, deployment_state):
        x = deployment_state.x
        edge_index = deployment_state.edge_index
        batch = deployment_state.batch
        x = self.conv(x, edge_index)  # 输出形状: [env.MS_AIMS_NUM, 2*hidden_channels]
        x = self.activate_func(x)
        x = self.conv2(x, edge_index)  # 输出形状: [env.MS_AIMS_NUM, hidden_channels]
        deployment_features = global_mean_pool(x, batch)
        return deployment_features

class Rout_Feature_Extractor(nn.Module):
    def __init__(self,env, args, hidden = 256,out_channels=8):
        super().__init__()
        self.conv = geom_nn.GATv2Conv(env.MS_AIMS_NUM, hidden,heads=2).to(env.device)
        self.conv2 = geom_nn.GATv2Conv(2*hidden, out_channels, heads=1).to(env.device)
        # self.lstm = nn.LSTM(input_size=hidden, hidden_size=out_channels, batch_first=True, num_layers=2).to(env.device) # 用于多子图节点特征聚合
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            init_gat_layer(self.conv)
            init_gat_layer(self.conv2)
            # init_lstm_layer(self.lstm)
        self.global_pool = global_mean_pool
    def forward(self,rout_state):
        x = rout_state.x
        edge_index = rout_state.edge_index
        batch = rout_state.batch
        x = self.conv(x, edge_index)  # 输出形状: [总节点数, 2*hidden_channels]
        x = self.activate_func(x)
        x = self.conv2(x, edge_index)  # 输出形状: [总节点数, hidden_channels]
        rout_subgraph_features = global_mean_pool(x, batch)
        rout_feature = torch.mean(rout_subgraph_features, dim=0).unsqueeze(0)
        # self.lstm.flatten_parameters()
        # rout_feature, _ = self.lstm(rout_subgraph_features.unsqueeze(0))  # 输入形状 [1, 30, out_channels]
        # rout_feature = rout_feature[:, -1, :]  # 取最后一个时间步输出，形状 [1, out_channels]
        return rout_feature

class MS_dependency_Feature_Extractor(nn.Module):
    def __init__(self,env, args, hidden = 256,out_channels=4):
        super().__init__()
        self.conv = geom_nn.GATv2Conv(6, hidden,heads=2).to(env.device)
        self.conv2 = geom_nn.GATv2Conv(2*hidden, out_channels, heads=1).to(env.device)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            init_gat_layer(self.conv)
            init_gat_layer(self.conv2)
        self.global_pool = global_mean_pool
    def forward(self, ms_dependency_state):
        x = ms_dependency_state.x
        edge_index = ms_dependency_state.edge_index
        batch = ms_dependency_state.batch
        x = self.conv(x, edge_index)  # 输出形状: [env.MS_AIMS_NUM, 2*hidden_channels]
        x = self.activate_func(x)
        x = self.conv2(x, edge_index)  # 输出形状: [env.MS_AIMS_NUM, hidden_channels]
        ms_dependencyfeatures = global_mean_pool(x, batch)
        return ms_dependencyfeatures

class MLP_Feature_Extractor(nn.Module):
    def __init__(self,env, args, hidden = 256):
        super().__init__()
        self.mlp_feature1 = nn.Linear(20, hidden).to(env.device)
        self.mlp_feature2 = nn.Linear(hidden, 10).to(env.device)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            orthogonal_init(self.mlp_feature1)
            orthogonal_init(self.mlp_feature2)
    def forward(self, mlp_input):
        x = self.mlp_feature1(mlp_input)
        x = self.activate_func(x)
        x = self.mlp_feature2(x)
        return x

class GNN_Actor(nn.Module):
    def __init__(self, env,args, action_dim):
        super().__init__()
        # 特征融合模块
        self.deployment_feature = Deployment_Feature_Extractor(env, args)
        self.rout_feature = Rout_Feature_Extractor(env, args)
        self.ms_dependency_feature = MS_dependency_Feature_Extractor(env, args)
        self.mlp_feature = MLP_Feature_Extractor(env, args)

        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(20+10, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)).to(env.device)

        if args.use_orthogonal_init:
            self.actor.apply(init_weights)
    def forward(self, state, ms_mark, env):
        # mlp_state, deployment, rout, ms_dependency = zip(*state)
        mlp_state = []
        batch = len(state)
        if batch==1:
            mlp_state.append(state[batch-1][0])
            deployment = state[batch-1][1]
            rout = state[batch-1][2]
            ms_dependency = state[batch-1][3]
            deployment_feature = self.deployment_feature(deployment)
            rout_feature = self.rout_feature(rout)
            ms_dependency_feature = self.ms_dependency_feature(ms_dependency)
            gnn_feature = torch.cat([deployment_feature,rout_feature,ms_dependency_feature], dim=1)
            # print(f"gnn_feature:{gnn_feature}")
        else:
            for idx in range(batch):
                mlp_state.append(state[idx][0])
            gnn_feature = torch.cat([torch.cat([self.deployment_feature(state[idx][1]),
                                                self.rout_feature(state[idx][2]),
                                                self.ms_dependency_feature(state[idx][3])],
                                               dim=1) for idx in range(batch)],
                                    dim=0)
        mlp_state = torch.tensor(mlp_state, dtype=torch.float32).to(env.device)
        mlp_input = self.mlp_feature(mlp_state)
        # print(f"mlp_input:{mlp_input}")
        a = self.actor(torch.cat([mlp_input,gnn_feature], dim=1))
        action_mask = torch.tensor(ms_mark, device=env.device) # 进行无效动作掩码
        s_masked_logits = a.masked_fill(action_mask == 0, -1e9)
        a_prob = torch.softmax(s_masked_logits, dim=1)
        return a_prob


class GNN_Critic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        # 特征融合模块
        # 特征融合模块
        self.deployment_feature = Deployment_Feature_Extractor(env, args)
        self.rout_feature = Rout_Feature_Extractor(env, args)
        self.ms_dependency_feature = MS_dependency_Feature_Extractor(env, args)
        self.mlp_feature = MLP_Feature_Extractor(env, args)
        self.critic = nn.Sequential(
            nn.Linear(20 + 10, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)).to(env.device)

        if args.use_orthogonal_init:
            self.critic.apply(init_weights)
    def forward(self, state, env):
        mlp_state = []
        batch = len(state)
        if batch == 1:
            mlp_state.append(state[batch - 1][0])
            deployment = state[batch - 1][1]
            rout = state[batch - 1][2]
            ms_dependency = state[batch - 1][3]
            deployment_feature = self.deployment_feature(deployment)
            rout_feature = self.rout_feature(rout)
            ms_dependency_feature = self.ms_dependency_feature(ms_dependency)
            gnn_feature = torch.cat([deployment_feature, rout_feature, ms_dependency_feature], dim=1)
        else:
            for idx in range(batch):
                mlp_state.append(state[idx][0])
            gnn_feature = torch.cat([torch.cat([self.deployment_feature(state[idx][1]),
                                                self.rout_feature(state[idx][2]),
                                                self.ms_dependency_feature(state[idx][3])],
                                               dim=1) for idx in range(batch)],
                                    dim=0)
        mlp_state = torch.tensor(mlp_state, dtype=torch.float32).to(env.device)
        mlp_input = self.mlp_feature(mlp_state)
        v = self.critic(torch.cat([mlp_input, gnn_feature], dim=1))
        return v


# DQN核心网络（单Q网络 + 目标网络）
# DQN核心网络（单Q网络 + 目标网络）
class DQN_QNetwork(nn.Module):
    def __init__(self, env, args, action_dim):
        super().__init__()
        # 复用PPO的特征提取模块
        self.deployment_feature = Deployment_Feature_Extractor(env, args)
        self.rout_feature = Rout_Feature_Extractor(env, args)
        self.ms_dependency_feature = MS_dependency_Feature_Extractor(env, args)
        self.mlp_feature = MLP_Feature_Extractor(env, args)

        # Q值输出层（修正输入维度为30）
        self.q_net = nn.Sequential(
            nn.Linear(10 + 8 + 8 + 4, 256),  # 30维输入
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        ).to(env.device)

        if args.use_orthogonal_init:
            self.q_net.apply(init_weights)

    def forward(self, state, env):
        batch = len(state)
        mlp_states = []
        deployment_states = []
        rout_states = []
        ms_dependency_states = []

        for s in state:
            mlp_states.append(s[0])
            deployment_states.append(s[1])
            rout_states.append(s[2])
            ms_dependency_states.append(s[3])

        # 提取GNN特征（单个样本处理）
        deployment_features = [self.deployment_feature(s) for s in deployment_states]
        rout_features = [self.rout_feature(s) for s in rout_states]
        ms_dependency_features = [self.ms_dependency_feature(s) for s in ms_dependency_states]

        # 拼接为批量张量
        mlp_input = torch.tensor(mlp_states, dtype=torch.float32, device=env.device)
        mlp_feat = self.mlp_feature(mlp_input)  # [batch, 10]

        deployment_feat = torch.cat(deployment_features, dim=0)  # [batch, 8]
        rout_feat = torch.cat(rout_features, dim=0)  # [batch, 8]
        ms_dependency_feat = torch.cat(ms_dependency_features, dim=0)  # [batch, 4]

        # 融合特征
        fused_feat = torch.cat([mlp_feat, deployment_feat, rout_feat, ms_dependency_feat], dim=1)  # [batch, 30]
        return self.q_net(fused_feat)


### **经验回放缓冲区**
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.idx = 0

    def store(self, state, action, reward, next_state, done, action_mask):
        data = (state, action, reward, next_state, done, action_mask)
        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
        else:
            self.buffer[self.idx] = data
            self.idx = (self.idx + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]