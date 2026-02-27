import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import network
from high_return_replay_buffer import High_Return_Replay_Buffer
from replay_buffer import ReplayBuffer

torch.autograd.set_detect_anomaly(True)  # 启用异常检测[5,6](@ref)


class EpsilonScheduler:
    def __init__(self, start=0.5, end=0.0, decay_steps=2000):
        self.epsilon = start
        self.start = start
        self.end = end
        self.decay = (start - end) / decay_steps

    def step(self):
        self.epsilon = max(self.end, self.epsilon)
        return self.epsilon

class PPOAgent:
    def __init__(self,env, args):
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_episodes = args.max_train_episodes
        self.max_episode_steps = args.max_episode_steps
        self.actor = network.GNN_Actor(env,args, self.action_dim).to(env.device)
        self.actor_target = network.GNN_Actor(env,args, self.action_dim).to(env.device)
        self.critic = network.GNN_Critic(env,args).to(env.device)
        self.critic_target = network.GNN_Critic(env,args).to(env.device)
        self.buffer = ReplayBuffer()
        self.SIL_buffer = High_Return_Replay_Buffer(args)

        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm


        # self.ACNet = network.ACNet(state_dim,action_dim)
        self.gamma = args.gamma
        self.lambad_ = args.lamda
        self.clip_range = args.clip_range
        self.sil_clip_range = args.SIL_clip_range
        self.entropy_coef = args.entropy_coef
        self.ppo_epochs = args.ppo_epochs
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        # self.ACNet_optimizer = optim.Adam(self.ACNet.parameters(), lr= 3e-4)


    def evaluate(self,state,ms_mark,env):
        action_probabilities = self.actor(state,ms_mark,env).detach().cpu().numpy().flatten()
        print([float('{:.4f}'.format(i)) for i in action_probabilities])
        action = np.argmax(action_probabilities)
        return action
    def get_action_epsilon(self, state, ms_mark, env, epsilon):
        action_probabilities = self.actor(state,ms_mark,env)
        index = []
        for idx in range(len(ms_mark)):
            if ms_mark[idx] ==1 :
                index.append(idx)
        with torch.no_grad():
            if np.random.rand() < epsilon:
                a = torch.tensor(np.array([np.random.choice(index)]), dtype=torch.int).to(env.device)
                dist = Categorical(action_probabilities)
                a_logprob = dist.log_prob(a)
            else:
                dist = Categorical(action_probabilities)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]
    def get_action(self, state, ms_mark, env):
        action_probabilities = self.actor(state,ms_mark,env)
        # print(action_probabilities)
        with torch.no_grad():
            dist = Categorical(action_probabilities)
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]
    def get_max_prob_action(self, state, ms_mark, env):
        action_probabilities = self.actor(state,ms_mark,env)
        # print(action_probabilities)
        with torch.no_grad():
            a = torch.argmax(action_probabilities, dim=-1)  # 获取最大概率的索引
            dist = Categorical(action_probabilities)
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]

    def update_model(self, env, args, total_steps):
        self.SIL_buffer.select_high_return_trajectory_data(args)
        self.SIL_buffer.clean()
        states, actions, old_log_probs, rewards, next_states, dw, dones, ms_marks = self.buffer.sample(env.device)

        advantage_list = []
        advantage = 0.0
        with torch.no_grad():
            v = self.critic(states, env)
            n_v = self.critic(next_states, env)
            td_target = rewards + self.gamma * n_v * (1 - dones)
            td_delta =  td_target- v
            for delta, d in zip(reversed(td_delta.flatten().cpu().numpy()), reversed(dones.flatten().cpu().numpy())):
                advantage = delta + self.gamma * self.lambad_ * advantage * (1.0 - d)
                advantage_list.insert(0, advantage)
            advantages = torch.tensor(advantage_list, dtype=torch.float).view(-1, 1).to(env.device)
            if self.use_adv_norm:  # Trick 1:advantage normalization
                advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-5)).to(env.device)

        policy_losses = np.zeros(self.ppo_epochs)
        value_losses = np.zeros(self.ppo_epochs)
        actor_grad_norms = np.zeros(self.ppo_epochs)
        critic_grad_norms = np.zeros(self.ppo_epochs)
        mean_entropy = 0
        # # PPO多epoch优化
        for p in range(self.ppo_epochs):
            min_batch_policy_loss = 0
            min_batch_value_loss = 0
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                # ppo
                prob = self.actor(states[index],ms_marks[index],env)
                new_dist = Categorical(probs=prob)
                entropy = new_dist.entropy().view(-1, 1).to(env.device)  # shape(mini_batch_size X 1)
                mean_entropy += torch.mean(entropy).item()
                new_log_prob = new_dist.log_prob(actions[index].squeeze()).view(-1, 1).to(env.device)
                # 计算比率
                ratio = torch.exp(new_log_prob - old_log_probs[index]).to(env.device)
                surr1 = ratio * advantages[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[index]  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2)-self.entropy_coef*entropy) # PPO损失函数

                v_s = self.critic(states[index], env)
                critic_loss = torch.mean(F.mse_loss(v_s, td_target[index].detach()))

                if args.use_self_imitation:
                    # self_imitation
                    SIL_s, SIL_a, SIL_r, SIL_mark = self.SIL_buffer.sample(env.device)
                    sil_prob = self.actor(SIL_s, SIL_mark, env)
                    sil_dist = Categorical(probs=sil_prob)
                    sil_log_prob = sil_dist.log_prob(SIL_a.squeeze()).view(-1, 1).to(env.device)
                    v_sil = self.critic(SIL_s, env)
                    sil_adg = torch.clamp(SIL_r - v_sil.detach(), min=0)
                    p_loss_sil = torch.mean(-sil_log_prob * sil_adg)
                    v_loss_sil = torch.mean(0.5 * sil_adg.pow(2))
                    total_actor_loss = actor_loss+self.sil_clip_range*p_loss_sil
                    total_critic_loss = critic_loss + self.sil_clip_range * v_loss_sil
                else:
                    total_actor_loss = actor_loss
                    total_critic_loss = critic_loss
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                if self.use_lr_decay:  # Trick 6:learning rate Decay
                    self.lr_decay(total_steps)

                # 统计损失变化
                min_batch_policy_loss += actor_loss
                min_batch_value_loss += critic_loss
                with torch.no_grad():
                    # 打印Actor/Critic参数的梯度范数
                    actor_grad_norm = 0.0
                    for param in self.actor.parameters():
                        if param.grad is not None:
                            actor_grad_norm += param.grad.norm().item()
                    actor_grad_norms[p] = actor_grad_norm
                    critic_grad_norm = 0.0
                    for param in self.critic.parameters():
                        if param.grad is not None:
                            critic_grad_norm += param.grad.norm().item()
                    critic_grad_norms[p] = critic_grad_norm
            policy_losses[p] = min_batch_policy_loss/(self.batch_size/self.mini_batch_size)
            value_losses[p] = min_batch_value_loss/(self.batch_size/self.mini_batch_size)
        print(f"Actor Grad Norm: {actor_grad_norms.sum()}, Critic Grad Norm: {critic_grad_norms.sum()}")
        return policy_losses.mean(), value_losses.mean(), mean_entropy/(self.ppo_epochs*3), sum(advantage_list)/len(advantage_list)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / (self.max_train_episodes * self.max_episode_steps))
        lr_c_now = self.lr_c * (1 - total_steps / (self.max_train_episodes * self.max_episode_steps))
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now

