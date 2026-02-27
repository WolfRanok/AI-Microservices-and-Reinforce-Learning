import torch
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.mark = []
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

    def sample(self, device):
        s = np.array(self.s, dtype=object)
        a = torch.tensor(np.array(self.a), dtype=torch.long).to(device)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(np.array(self.a_logprob), dtype=torch.float).to(device)
        r = torch.tensor(np.array(self.r), dtype=torch.float).to(device)
        s_ = np.array(self.s_, dtype=object)
        dw = torch.tensor(np.array(self.dw), dtype=torch.float).to(device)
        done = torch.tensor(np.array(self.done), dtype=torch.float).to(device)
        mark = np.array(self.mark)
        return s, a, a_logprob, r, s_, dw, done, mark

    def clean(self):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.mark = []
