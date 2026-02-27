import itertools
import random

import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class High_Return_Replay_Buffer:
    def __init__(self, args):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.mark = []
        self.max_r = 0
        self.high_return_s = []
        self.high_return_a = []
        self.high_return_r = []
        self.high_return_mark = []
        self.freshness = 20
        self.gamma = args.gamma
        self.high_return_size = args.high_return_buffer_size
        self.sample_batch_size = args.SIL_sample_batch_size
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done, mark):
        self.s.append(s)
        self.a.append([a])
        self.a_logprob.append([a_logprob])
        self.r.append([r])
        self.s_.append(s_)
        self.dw.append([dw])
        self.done.append([done])
        self.mark.append(mark)

    def clean(self):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.mark = []

    def select_high_return_trajectory_data(self,args):
        # calculate the return on action
        returns = []
        return_ = 0
        for r, d in zip(reversed(self.r), reversed(self.done)):
            return_ = r[0] + self.gamma * return_ * (1 - d[0])
            returns.insert(0, [return_])
        self.r = returns.copy()
        max_return = 0
        # finding the highest return on action
        for r in self.r:
            max_return = max(max_return, r[0])
        # collect high-return actions
        rate = 0
        each_episode_s = [self.s[i:i + args.mini_batch_size] for i in range(0, len(self.s), args.mini_batch_size)]
        each_episode_a = [self.a[i:i + args.mini_batch_size] for i in range(0, len(self.a), args.mini_batch_size)]
        each_episode_r = [self.r[i:i + args.mini_batch_size] for i in range(0, len(self.r), args.mini_batch_size)]
        each_episode_mark = [self.mark[i:i+args.mini_batch_size] for i in range(0, len(self.mark), args.mini_batch_size)]
        r = []
        for idx in range(int(args.batch_size/args.mini_batch_size)):
            r.append(sum(itertools.chain.from_iterable(each_episode_r[idx])))
        index_batch = sorted(range(int(args.batch_size/args.mini_batch_size)), key=lambda k: r[k], reverse=False)
        for episode_idx in index_batch:
            c_s = each_episode_s[episode_idx]
            c_a = each_episode_a[episode_idx]
            c_r = each_episode_r[episode_idx]
            c_mark = each_episode_mark[episode_idx]
            total_r = r[episode_idx]
            # if total_r <= self.max_r:
            #     self.freshness -= 1
            # if self.freshness==0:
            #     self.max_r = 0
            #     self.freshness = 20
            if total_r >= self.max_r*0.90:
                if total_r>self.max_r:
                    self.max_r = total_r
                for idx in range(args.mini_batch_size):
                    if c_r[idx] > max_return * rate:
                        if len(self.high_return_r) <= self.high_return_size:
                            self.high_return_s.append(c_s[idx])
                            self.high_return_a.append(c_a[idx])
                            self.high_return_r.append(c_r[idx])
                            self.high_return_mark.append(c_mark[idx])
                            self.count += 1
                        else:
                            # 按照时间推移替换高回报经验
                            self.high_return_s[self.count % self.high_return_size] = c_s[idx]
                            self.high_return_a[self.count % self.high_return_size] = c_a[idx]
                            self.high_return_r[self.count % self.high_return_size] = c_r[idx]
                            self.high_return_mark[self.count % self.high_return_size] = c_mark[idx]
                            self.count += 1
                            # # 按照回报大小替换高回报经验
                            # outdated_data_idx = self.high_return_r.index(min(self.high_return_r))
                            # if self.high_return_r[outdated_data_idx] < self.r[idx]:
                            #     self.high_return_s[outdated_data_idx] = self.s[idx]
                            #     self.high_return_a[outdated_data_idx] = self.a[idx]
                            #     self.high_return_r[outdated_data_idx] = self.r[idx]
                            #     self.high_return_mark[outdated_data_idx] = self.mark[idx]



    def sample(self, device):
        batch_index = random.sample(range(len(self.high_return_r)), self.sample_batch_size)
        s = np.array(self.high_return_s, dtype=object)
        a = torch.tensor(np.array(self.high_return_a), dtype=torch.long).to(device)  # In discrete action space, 'a' needs to be torch.long
        r = torch.tensor(np.array(self.high_return_r), dtype=torch.float).to(device)
        mark = np.array(self.high_return_mark)
        return s[batch_index], a[batch_index], r[batch_index], mark[batch_index]
