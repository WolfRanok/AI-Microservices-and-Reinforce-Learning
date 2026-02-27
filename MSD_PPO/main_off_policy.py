import csv
import Agent_off_policy
from Interaction import *
from ENV import *

state_dim=MA_AIMS_NUM * NODE_NUM \
          + 3 * 2 * NODE_NUM \
          + 8\
          + NODE_NUM*4\
          + MA_AIMS_NUM * MA_AIMS_NUM \
          + NODE_NUM * NODE_NUM
action_dim = NODE_NUM
step_size = 2

if __name__ == '__main__':

    Max_EPISODES = 1000000
    Max_STEP = 100
    agent = Agent_off_policy.PPOAgent(state_dim, action_dim)
    for episode in range(Max_EPISODES):
        if episode==299:
            a=1
        state = initial_state()
        env = environment_interaction_ms_initial()
        is_done = False
        episode_reward = 0
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        while not is_done:
            env.dep_ms_count += 1
            action = agent.ACNet_get_action(torch.tensor(state, dtype=torch.float32))  # 行动
            env.index = ms_index = env.option_ms()  # 选择需要部署的类型
            next_state = env.get_next_state(ms_index, state, action)  # 状态
            is_done = False
            # 计算奖励值
            if env.is_it_sufficient(ms_index, state, action):  # 可以分配
                if env.is_it_over():  # 部署结束
                    reward = env.get_reward(0, ms_index, state, next_state)
                    is_done = True
                else:  # 部署未结束
                    reward = env.get_reward(1, ms_index, state, next_state)
            else:  # 不能分配
                if env.is_it_over():
                    is_done = True
                reward = env.get_reward(-1, ms_index, state, next_state)
            episode_reward += reward
            states.append(torch.tensor(state, dtype=torch.float32))
            actions.append(action)
            rewards.append(reward)
            next_states.append(torch.tensor(next_state, dtype=torch.float32))
            dones.append(is_done)
            state = next_state.copy()
        returns = agent.compute_returns(rewards,dones)

        advantages = agent.compute_gae(states,rewards,dones)
        # advantages = agent.compute_gae_new(states,rewards,dones)

        for t in range(len(states)):
            agent.buffer.push(state=states[t], action=actions[t], reward=rewards[t], next_state=next_states[t],
                              done=dones[t], returns=returns[t], advantages=advantages[t])
        if agent.buffer.__len__() > 100:
            p_loss, v_loss, t_loss = agent.update_model(batch_size=64)
            print(f"episode:{episode}, episode_reward:{episode_reward}, p_loss:{p_loss}, v_loss:{v_loss}, t_loss:{t_loss}")
            agent.buffer.clear()

            with open('PPO_DATA_0318', 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # 第一次写入表头
                if file.tell() == 0:  # 检查文件是否为空
                    data = ["episode", "episode_reward", "p_loss", "v_loss", "t_loss"]
                    writer.writerow(data)
                # 写入新行
                new_row = [episode,episode_reward,p_loss,v_loss,t_loss]
                writer.writerow(new_row)




