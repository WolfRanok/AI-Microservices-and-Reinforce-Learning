# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import json
import os.path
from collections import deque

import torch
import torch.optim as optim
import csv
from Agent import *
from CY_Environment_Interaction import *

def save_model(Agent, statistics, data_list):
    """
    用于保存模型，由于不同模型接受的参数规模不一样（这是由于服务器数量，微服务类型数量等因素导致的），所以这里按照输入的类型进行命名
    :return:None
    """
    try:
        # 创建目录（如果不存在）
        os.makedirs("Model", exist_ok=True)
        os.makedirs("Data", exist_ok=True)
        # 保存模型
        torch.save(Agent.actor_target.state_dict(), f"Model/2025_02_27_actor_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}_new_state.pth")
        torch.save(Agent.critic_target.state_dict(), f"Model/2025_02_27_critic_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}_new_state.pth")

        # 保存数据
        with open('Data/2025_02_27_statistics_new_state.json', 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=4)
        with open(f'Data/2025_02_27_data_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}_new_state.csv', mode='w', newline="") as f:
            writer = csv.writer(f)
            # 写入标题
            writer.writerow(data_list.keys())
            # 写入行
            rows = zip(*data_list.values())
            writer.writerows(rows)
        print("模型已保存！")
    except FileNotFoundError as e:
        print(f"保存路径错误: {e}. 请检查目录是否正确。")
    except PermissionError as e:
        print(f"权限错误: {e}. 请检查文件写入权限。")
    except TypeError as e:
        print(f"数据序列化错误: {e}. 请确保 self.statistics 中的数据可被 JSON 序列化。")
    except Exception as e:
        print(f"保存模型和数据时发生未知错误: {e}")
def load_model(Agent):
    """
    当模型存在时，用于加载模型
    :return: None
    """

    actor_url = f"Model/2025_02_27_actor_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}_new_state.pth"
    critic_url = f"Model/2025_02_27_critic_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}_new_state.pth"

    try:
        if os.path.exists(actor_url) and os.path.exists(critic_url):
            Agent.actor_target.load_state_dict(torch.load(actor_url))
            Agent.critic_target.load_state_dict(torch.load(critic_url))

            # 网络复制
            Agent.actor.load_state_dict(Agent.actor_target.state_dict())
            Agent.critic.load_state_dict(Agent.critic_target.state_dict())
            print("模型成功加载")
        else:
            print("模型文件不存在，请检查路径")
    except Exception as e:
        print(f"加载模型时出错: {e}")

def train():
    SAVE_COUNT = 100
    MAX_EPISODES = 100000
    Agent = LSTM_DDPG_Agent(MA_AIMS_NUM, NODE_NUM, USER_NUM)
    environment_interaction = environment_interaction_ms_initial()
    old_episode_count = 0
    statistics = []
    data_list = {  # 数据字典
        'episode': [],
        'sum_reward': [],
        'loss': [],
        'param_change': [],
        'T': [],
    }
    rewards = 0
    for episode in range(1,MAX_EPISODES+1):
        state = initial_state()  # 初始化状态
        environment_interaction.refresh()  # 更新镜像需求数
        batch_size = int(environment_interaction.ms_image.sum())
        episode_count = 0  # 记录当前迭代的长度
        fail_count = 0  # 记录失败次数
        sum_fail_count = 0  # 记录未能部署上的节点数目
        data = {  # 数据字典
            'episode': episode + old_episode_count,
            'sum_reward': 0,
            'loss': 0,
            'param_change': 0,
            'T': 0,
        }
        reward_list = []
        while True:
            if episode > 10:
                a = 1
            # 产生动作，和下一个状态
            environment_interaction.dep_ms_count += 1

            action, _ = Agent.actor(torch.tensor(state, dtype=torch.float32))  # 由actor产生动作
            environment_interaction.index = ms_index = environment_interaction.option_ms()  # 选择需要部署的类型
            action_index = environment_interaction.get_action(ms_index, action)  # 行动
            # print(index, action_probabilities[index] , action)

            next_state = environment_interaction.get_next_state(ms_index, state, action_index)  # 状态
            # print(state==next_state)
            is_valid = False
            # 计算奖励值
            if environment_interaction.is_it_sufficient(ms_index, state, action_index):  # 可以分配
                is_valid = True
                if environment_interaction.is_it_over():  # 部署结束
                    reward = environment_interaction.get_reward(0, ms_index, state, next_state, episode_count)
                else:  # 部署未结束
                    reward = environment_interaction.get_reward(1, ms_index, state, next_state, episode_count)
            else:  # 不能分配
                reward = environment_interaction.get_reward(-1, ms_index, state, next_state, episode_count)
            if is_valid:
                data['sum_reward'] += reward
                reward_list.append(reward)
            else:
                # environment_interaction.pass_round(ms_index, state)  # 跳过当前部署
                sum_fail_count += 1
                data['sum_reward'] += reward
                reward_list.append(reward)
            done = 0
            if environment_interaction.is_it_over():
                done = 1
            # 执行训练模型的训练
            Agent.buffer.push(state, action, reward, next_state, done)
            if len(Agent.buffer) > batch_size:
                Agent.update_model(batch_size, data)
                Agent.buffer.clean_buffer()
            ## 数据更新
            episode_count += 1  # 记录训练次数
            state = next_state.copy()  # 更新状态
            # 部署结束退出循环
            if done:
                break
        print(get_deploy(state))
        print(get_resource(state))
        data['T'] = environment_interaction.get_T(state)
        statistics.append(data)  # 记录训练数据
        data_list['episode'].append(data['episode'])
        data_list['sum_reward'].append(data['sum_reward'])
        data_list['loss'].append(data['loss'])
        data_list['param_change'].append(data['param_change'])
        data_list['T'].append(data['T'])
        if episode == 2:
            rewards = data['sum_reward']
        elif episode != 1:
            rewards = rewards * 0.95 + data['sum_reward'] * 0.05

        # print(f"第 {episode} 次的单次部署的最小时延变化为：{self.environment_interaction.T_min_list}")
        # print(f"第 {episode} 次的单次部署的时延变化为：{self.environment_interaction.T_list}")
        print(f"第 {episode} 次的奖励分布：{reward_list}")
        # print(f"第 {episode} 次的微服务部署情况：{get_deploy(state)}")
        # print(f"{is_f}")
        num = 0
        for i in reward_list:
            if i == 0:
                num += 1
        print(num)
        print(num == sum_fail_count)
        print(
            f"第 {episode} 次迭代执行了 {episode_count} 次训练, 当前部署得到的延迟为 {data['T']}，函数损失值loss为 {data['loss']} ，一共有 {environment_interaction.sum_ms_aims} 个待部署实例，其中有 {sum_fail_count} 个实例没有部署上",
            {rewards})
        print(data)
        print(cal_load_balance(state))
        # print(self.data_list)

        # 指定一段时间保存一次模型
        if episode % SAVE_COUNT == 0:
            save_model(Agent, statistics, data_list)
        # self.environment_interaction.analysis_state(state)

    print("训练完成")


def re():
    """
    这是用来做推理的一个函数，不是训练
    :return:
    """
    Agent = LSTM_DDPG_Agent(MA_AIMS_NUM, NODE_NUM, USER_NUM)
    load_model(Agent)
    environment_interaction = environment_interaction_ms_initial()
    state = initial_state()  # 初始化状态
    while True:
        # 产生动作，和下一个状态
        environment_interaction.dep_ms_count += 1
        environment_interaction.index = this_index = environment_interaction.option_ms()  # 选择需要部署的类型

        action = Agent.actor(torch.tensor(state, dtype=torch.float32))  # 由actor产生动作
        print(action)
        action_index = environment_interaction.get_action(this_index, action)  # 行动
        # print(index, action_probabilities[index] , action)
        next_state = environment_interaction.get_next_state(this_index, state, action_index)  # 状态
        state = next_state.copy()
        if environment_interaction.is_it_over():
            break
    environment_interaction.analysis_state(state=state)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    train()
    re()




# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
