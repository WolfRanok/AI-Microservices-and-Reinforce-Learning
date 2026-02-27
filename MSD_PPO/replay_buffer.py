import random
from typing import Tuple, List, Any

import numpy as np
import torch
from collections import deque

from torch import Tensor, FloatTensor


class ReplayBuffer:
    """基础FIFO回放缓冲区，支持随机批量采样"""

    def __init__(self, max_capacity: int):
        """
        初始化回放缓冲区
        :param max_capacity: 最大存储容量
        """
        self.max_capacity = max_capacity
        self.buffer = deque(maxlen=max_capacity)  # 自动限制容量

    def push(self, state, action, reward, next_state, done, returns, advantages):
        """添加新经验到缓冲区"""
        self.buffer.append((
            torch.FloatTensor(state),
            torch.FloatTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.FloatTensor([done]),
            torch.FloatTensor([returns]),
            torch.FloatTensor([advantages])
        ))


    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """从缓冲区中随机采样一批数据
        :param batch_size: 采样数量
        :return: 采样的经验元组列表
        """
        if len(self.buffer) < batch_size:
            raise ValueError("Sample size exceeds buffer size")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, returns, advantages = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones),
            torch.stack(returns),
            torch.stack(advantages)
        )

    def __len__(self):
        return len(self.buffer)
    def clear(self):
        self.buffer.clear()

if __name__ == '__main__':
    # 初始化缓冲区（最大容量10000）
    buffer = ReplayBuffer(max_capacity=10000)

    # 存储经验（例如在训练循环中）
    for _ in range(10):
        state = [1]  # 当前状态
        action = [2]  # 采取的动作
        reward = 1  # 获得奖励
        next_state = [11]  # 下一个状态
        done = 0  # 是否完成
        buffer.push(state, action, reward, next_state, done)

    # 随机采样一批数据（batch_size=64）
    states, actions, rewards, next_states, dones = zip(*buffer.sample(4))
    print(states, actions, rewards, next_states, dones)
    print(buffer.__len__())