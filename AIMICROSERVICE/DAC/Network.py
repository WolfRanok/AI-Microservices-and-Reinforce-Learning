import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from Environment.NEW_ENV import *
MA_AIMS_NUM = MS_NUM + AIMS_NUM

# class Actor(nn.Module):
#     def __init__(self, ma_aims_num, node_num, user_num):
#         super(Actor, self).__init__()
#         self.ma_aims_num = ma_aims_num
#         self.node_num = node_num
#
#         # 输入维度规模
#         input_size = ma_aims_num * node_num \
#                      + 3 * 2 * node_num \
#                      + user_num*(node_num*node_num+node_num) \
#                      + USER_NUM*MA_AIMS_NUM \
#                      + node_num*node_num\
#                      + user_num*(4+ma_aims_num) \
#                      + node_num*3
#
#         # 输出维度
#         output_size = ma_aims_num * node_num
#
#         # LSTM层
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
#
#         # 全连接层
#         self.fc1 = nn.Linear(128, 256)
#         self.fc2 = nn.Linear(256, output_size)
#
#         # 初始化权重
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#
#     def forward(self, inputs):
#         # 转化输入为torch.Tensor对象
#         if not isinstance(inputs, torch.Tensor):
#             x = torch.tensor(inputs, dtype=torch.float32)
#         else:
#             x = inputs
#
#         # 输入需要增加时间步维度和批量维度，形状变为 (batch_size=1, seq_len=1, input_size)
#         x = x.unsqueeze(0).unsqueeze(0)
#
#         # LSTM 前向传播
#         lstm_out, _ = self.lstm(x)  # 输出形状为 (batch_size=1, seq_len=1, hidden_size=128)
#         x = lstm_out.squeeze(0).squeeze(0)  # 取出张量，仅保留 (hidden_size=128)
#
#         # 全连接层
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#
#         # 形状调整为 (MA_AIMS_NUM, NODE_NUM)
#         x = x.view(self.ma_aims_num, self.node_num)
#
#         # Softmax 处理，每行表示概率分布
#         probabilities = F.softmax(x, dim=1)
#
#         return probabilities
# class Critic(nn.Module):
#     def __init__(self, ma_aims_num, node_num, user_num):
#         super(Critic, self).__init__()
#         self.ma_aims_num = ma_aims_num
#         self.node_num = node_num
#
#         # 输入维度
#         input_size = ma_aims_num*node_num \
#                      + 3*2*node_num \
#                      + user_num*(node_num*node_num+node_num) \
#                      + USER_NUM * MA_AIMS_NUM \
#                      + node_num * node_num \
#                      + user_num * (4 + ma_aims_num) \
#                      + node_num * 3 \
#                      + ma_aims_num*node_num
#
#         # LSTM层
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
#
#         # 全连接层
#         self.fc1 = nn.Linear(128, 256)
#         self.fc2 = nn.Linear(256, 1)  # 输出为单个评价值
#
#         # 初始化权重
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#
#     def flatter(self, state, action):
#         """
#         将状态和动作合并，并且一维化
#         :param state: state
#         :param action: action
#         :return:
#         """
#         # 转化输入为torch.Tensor对象
#         if not isinstance(state, torch.Tensor):
#             state = torch.tensor(state, dtype=torch.float32)
#         if not isinstance(action, torch.Tensor):
#             action = torch.tensor(action, dtype=torch.float32)
#         # 改变形状
#         action = action.view(-1)
#         return torch.cat((state, action), dim=0)
#
#     def forward(self, state, action):
#
#         inputs = self.flatter(state, action)
#
#         # 输入需要增加时间步维度和批量维度，形状变为 (batch_size=1, seq_len=1, input_size)
#         x = inputs.unsqueeze(0).unsqueeze(0)
#
#         # LSTM 前向传播
#         lstm_out, _ = self.lstm(x)  # 输出形状为 (batch_size=1, seq_len=1, hidden_size=128)
#         x = lstm_out.squeeze(0).squeeze(0)  # 取出张量，仅保留 (hidden_size=128)
#
#         # 全连接层
#         x = F.relu(self.fc1(x))
#         value = self.fc2(x)
#
#         return value
class Actor(nn.Module):
    def __init__(self, ma_aims_num, node_num, user_num):
        super(Actor, self).__init__()
        self.ma_aims_num = ma_aims_num
        self.node_num = node_num

        # 输入维度规模
        # input_size = ma_aims_num * node_num \
        #              + 3 * 2 * node_num \
        #              + user_num*(node_num+1)*node_num*ma_aims_num \
        #              + user_num*ma_aims_num \
        #              + node_num*node_num\
        #              + user_num*(4+ma_aims_num) \
        #              + node_num*3
        input_size = ma_aims_num * node_num \
                     + 3 * 2 * node_num \
                     + user_num * (node_num + 1) * node_num * ma_aims_num \
                     + user_num * ma_aims_num \
                     + 3 * ma_aims_num \
                     + ma_aims_num * ma_aims_num \
                     + node_num * node_num
        # 输出维度
        output_size = ma_aims_num * node_num

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, output_size)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, inputs):
        # 转化输入为torch.Tensor对象
        if not isinstance(inputs, torch.Tensor):
            x = torch.tensor(inputs, dtype=torch.float32)
        else:
            x = inputs

        # 输入需要增加时间步维度和批量维度，形状变为 (batch_size=1, seq_len=1, input_size)
        x = x.unsqueeze(0).unsqueeze(0)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(x)  # 输出形状为 (batch_size=1, seq_len=1, hidden_size=128)
        x = lstm_out.squeeze(0).squeeze(0)  # 取出张量，仅保留 (hidden_size=128)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # 形状调整为 (MA_AIMS_NUM, NODE_NUM)
        x = x.view(self.ma_aims_num,self.node_num)

        # Softmax 处理，每行表示概率分布
        probabilities = F.softmax(x, dim=1)

        return probabilities
class Critic(nn.Module):
    def __init__(self, ma_aims_num, node_num, user_num):
        super(Critic, self).__init__()
        self.ma_aims_num = ma_aims_num
        self.node_num = node_num

        # 输入维度
        input_size = ma_aims_num * node_num \
                     + 3 * 2 * node_num \
                     + user_num * (node_num + 1) * node_num * ma_aims_num \
                     + user_num * ma_aims_num \
                     + 3 * ma_aims_num \
                     + ma_aims_num * ma_aims_num \
                     + node_num * node_num \
                     + ma_aims_num * node_num

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)  # 输出为单个评价值

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def flatter(self, state, action):
        """
        将状态和动作合并，并且一维化
        :param state: state
        :param action: action
        :return:
        """
        # 转化输入为torch.Tensor对象
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        # 改变形状
        action = action.view(-1)
        return torch.cat((state, action), dim=0)

    def forward(self, state, action):

        inputs = self.flatter(state, action)

        # 输入需要增加时间步维度和批量维度，形状变为 (batch_size=1, seq_len=1, input_size)
        x = inputs.unsqueeze(0).unsqueeze(0)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(x)  # 输出形状为 (batch_size=1, seq_len=1, hidden_size=128)
        x = lstm_out.squeeze(0).squeeze(0)  # 取出张量，仅保留 (hidden_size=128)

        # 全连接层
        x = F.relu(self.fc1(x))
        value = self.fc2(x)

        return value


class LSTMActor(nn.Module):
    def __init__(self, ms_aims_num, node_num, user_num, hidden_size=128, lstm_layers=1):
        super(LSTMActor, self).__init__()
        self.ms_aims_num = ms_aims_num
        self.node_num = node_num
        self.state_dim = ms_aims_num * node_num \
                     + 3 * 2 * node_num \
                     + user_num * (node_num + 1) * node_num * ms_aims_num \
                     + user_num * ms_aims_num \
                     + 3 * ms_aims_num \
                     + ms_aims_num * ms_aims_num \
                     + node_num * node_num
        self.action_dim = node_num
        self.lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),  # 输出服务器偏好分数
            # nn.Softmax(dim=-1)
        )
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

    def forward(self, state, hidden=None):
        state = state[np.newaxis, np.newaxis, :]
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        # state shape: (batch, seq_len, state_dim)
        batch_size = state.size(0)
        if hidden is None:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(state.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(state.device)
            hidden = (h0, c0)

        out, hidden = self.lstm(state, hidden)
        # 取序列最后一个时间步的输出
        out = out[:, -1, :]
        # return self.fc(out), hidden
        output = self.fc(out)
        # output = output.view(-1, self.ms_aims_num, self.node_num)  # Reshape 为二维矩阵
        output = torch.softmax(output, dim=1)  # 对每行应用 Softmax
        return output

class LSTMCritic(nn.Module):
    def __init__(self, ms_aims_num, node_num, user_num, hidden_size=128, lstm_layers=1):
        super(LSTMCritic, self).__init__()
        # 状态处理分支
        self.state_dim = ms_aims_num * node_num \
                         + 3 * 2 * node_num \
                         + user_num * (node_num + 1) * node_num * ms_aims_num \
                         + user_num * ms_aims_num \
                         + 3 * ms_aims_num \
                         + ms_aims_num * ms_aims_num \
                         + node_num * node_num
        self.action_dim = node_num
        self.state_lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        # 动作处理分支
        self.action_fc = nn.Linear(self.action_dim, hidden_size)

        # 联合处理层
        self.joint_fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出Q值
        )
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

    def forward(self, state, action, hidden=None):
        state = state[np.newaxis, np.newaxis, :]
        # action = action.view(-1)
        # action = action[np.newaxis,:]
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        # state shape: (batch, seq_len, state_dim)
        # action shape: (batch, action_dim)
        batch_size = state.size(0)

        # 处理状态序列
        if hidden is None:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(state.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(state.device)
            hidden = (h0, c0)
        state_out, hidden = self.state_lstm(state, hidden)
        state_feat = state_out[:, -1, :]  # (batch, hidden)

        # 处理动作
        action_feat = self.action_fc(action)  # (batch, hidden)

        # 联合特征
        joint = torch.cat([state_feat, action_feat], dim=-1)
        # return self.joint_fc(joint), hidden
        return self.joint_fc(joint)

# 示例代码
if __name__ == "__main__":

    # 输入数据示例
    example_input = initial_state()  # 创建一个输入数据
    print("输入状态数据如下")
    print(example_input)

    # 初始化 Actor 和 Critic 网络
    # actor = Actor(MA_AIMS_NUM, NODE_NUM, USER_NUM)
    # critic = Critic(MA_AIMS_NUM, NODE_NUM, USER_NUM)

    actor =  LSTMActor(MA_AIMS_NUM, NODE_NUM, USER_NUM)
    critic = LSTMCritic(MA_AIMS_NUM, NODE_NUM, USER_NUM)

    # 前向传播
    action_probabilities = actor(example_input)
    action_value = critic(example_input, action_probabilities)
    # print(f"隐藏层层数分别为:actor {ha},critic {hc}")

    print("actor网络的行动输入如下所示(Actor Output):")
    print(action_probabilities)
    print("\ncritic网络对该行动给出的评价 (Critic Output):")
    print(action_value)
"""
以下为一个输出示例：
输入状态数据如下
[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.  21.  21.   0.   0.   2.   4.   0.   0. 313. 371.]
actor网络的行动输入如下所示(Actor Output):
tensor([[0.7320, 0.2680],
        [0.5053, 0.4947],
        [0.5723, 0.4277],
        [0.5072, 0.4928],
        [0.5295, 0.4705],
        [0.5306, 0.4694],
        [0.3449, 0.6551],
        [0.4054, 0.5946]], grad_fn=<SoftmaxBackward0>)

critic网络对该行动给出的评价 (Critic Output):
tensor([0.4896], grad_fn=<ViewBackward0>)
"""