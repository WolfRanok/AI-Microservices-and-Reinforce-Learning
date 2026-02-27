import csv
import os
import wandb

import Agent_on_policy
from Interaction import *
from ENV import *
from Pre_training_data_acquisition import *


# state_dim=MA_AIMS_NUM * NODE_NUM \
#           + 3 * 2 * NODE_NUM \
#           + 8\
#           + NODE_NUM*4\
#           + MA_AIMS_NUM * MA_AIMS_NUM \
#           + NODE_NUM * NODE_NUM
state_dim=9+MA_AIMS_NUM+MA_AIMS_NUM*NODE_NUM*2+2*RESOURCE*NODE_NUM+2*NODE_NUM+MA_AIMS_NUM*MA_AIMS_NUM+NODE_NUM*NODE_NUM
action_dim = NODE_NUM
step_size = 2

def save_mode(Agent):
    """
        用于保存模型，由于不同模型接受的参数规模不一样（这是由于服务器数量，微服务类型数量等因素导致的），所以这里按照输入的类型进行命名
        :return:None
        """
    try:
        # 创建目录（如果不存在）
        os.makedirs("Model", exist_ok=True)
        # 保存模型
        torch.save(Agent.actor.state_dict(),
                   f"Model/2000_PPO_ON_POLICY_2025_03_24_actor_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}.pth")
        torch.save(Agent.critic.state_dict(),
                   f"Model/2000_PPO_ON_POLICY_2025_03_24_critic_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}.pth")
        print("模型已保存！")
    except FileNotFoundError as e:
        print(f"保存路径错误: {e}. 请检查目录是否正确。")
    except PermissionError as e:
        print(f"权限错误: {e}. 请检查文件写入权限。")
    except TypeError as e:
        print(f"数据序列化错误: {e}. 请确保 self.statistics 中的数据可被 JSON 序列化。")
    except Exception as e:
        print(f"保存模型和数据时发生未知错误: {e}")

def pre_training(pre_episode):
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    pre_deploy,_,_=get_random_deploy_dependency(transition_dict)
    pre_rout = get_each_request_rout(pre_deploy)
    pre_delay = cal_total_delay(pre_deploy, pre_rout)
    p_loss, v_loss, t_loss = agent.update_model(transition_dict)
    print(f"pre_episode:{pre_episode}, reward:{sum(transition_dict['rewards'])}, p_loss:{p_loss}, v_loss:{v_loss}, t_loss:{t_loss}, delay:{pre_delay}")


if __name__ == '__main__':
    run = wandb.init(project="ppo_data",name="2000_PPO_REWARD_LOSS",dir="MSD_PPO",job_type="training",reinit=True)
    MAX_PRE = 200
    Max_EPISODES = 1500
    Max_STEP = 100
    agent = Agent_on_policy.PPOAgent(state_dim, action_dim)
    # for pre_episode in range(MAX_PRE):
    #     pre_training(pre_episode)
    for episode in range(Max_EPISODES):
        state = init_state()
        env = environment_interaction_ms_initial()
        is_done = False
        episode_reward = 0
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        while not is_done:
            env.dep_ms_count += 1
            action = agent.get_action(torch.tensor(state, dtype=torch.float32).to(agent.device))  # 行动
            env.index = ms_index = env.option_ms()  # 选择需要部署的类型
            next_state = env.get_next_state_for_new_state(ms_index, state, action)  # 状态
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
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(is_done)
            state = next_state.copy()
        p_loss, v_loss, t_loss = agent.update_model(transition_dict)   # 这里会进行训练
        f_deploy = get_deploy_for_new_state(state)
        f_rout = get_each_request_rout(f_deploy)
        f_delay = cal_total_delay(f_deploy,f_rout)
        run.log({"actor_loss": p_loss, "critic_loss": v_loss, "rewards":episode_reward, "delay":f_delay})
        print(f"episode:{episode}, episode_reward:{episode_reward}, p_loss:{p_loss}, v_loss:{v_loss}, t_loss:{t_loss}, delay:{f_delay}")
        if episode%50==0:
            save_mode(Agent=agent)
        os.makedirs("Data", exist_ok=True)
        with open('Data/PPO_DATA_0329', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 第一次写入表头
            if file.tell() == 0:  # 检查文件是否为空
                data = ["episode", "episode_reward", "p_loss", "v_loss", "t_loss"]
                writer.writerow(data)
            # 写入新行
            new_row = [episode, episode_reward, p_loss, v_loss, t_loss]
            writer.writerow(new_row)





