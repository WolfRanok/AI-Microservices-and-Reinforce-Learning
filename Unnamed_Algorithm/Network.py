import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from Environment.ENV import *
MA_AIMS_NUM = MS_NUM + AIMS_NUM

class Actor(nn.Module):
    def __init__(self, ma_aims_num, node_num):
        super(Actor, self).__init__()
        self.ma_aims_num = ma_aims_num
        self.node_num = node_num

        # 输入维度规模
        input_size = ma_aims_num * node_num + 3 * 2 * node_num
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
        x = x.view(self.ma_aims_num, self.node_num)

        # Softmax 处理，每行表示概率分布
        probabilities = F.softmax(x, dim=1)

        return probabilities


class Critic(nn.Module):
    def __init__(self, ma_aims_num, node_num):
        super(Critic, self).__init__()
        self.ma_aims_num = ma_aims_num
        self.node_num = node_num

        # 输入维度
        input_size = ma_aims_num * node_num + 3 * 2 * node_num + ma_aims_num * node_num

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


# 示例代码
if __name__ == "__main__":

    # 输入数据示例
    example_input = initial_state()  # 创建一个输入数据
    print("输入状态数据如下")
    print(example_input)

    # 初始化 Actor 和 Critic 网络
    actor = Actor(MA_AIMS_NUM, NODE_NUM)
    critic = Critic(MA_AIMS_NUM, NODE_NUM)

    # 前向传播
    action_probabilities = actor(example_input)
    action_value = critic(example_input, action_probabilities)

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