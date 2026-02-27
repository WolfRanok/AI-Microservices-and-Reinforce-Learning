import csv
import os

import Agent_on_policy
from Interaction import *
from ENV import *


state_dim=9+MA_AIMS_NUM+MA_AIMS_NUM*NODE_NUM*2+2*RESOURCE*NODE_NUM+2*NODE_NUM+MA_AIMS_NUM*MA_AIMS_NUM+NODE_NUM*NODE_NUM
action_dim = NODE_NUM
step_size = 2

def load_model(agent):
    """
    当模型存在时，用于加载模型
    :return: None
    """

    actor_url = f"Model/2000_PPO_ON_POLICY_2025_03_24_actor_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}.pth"
    critic_url = f"Model/2000_PPO_ON_POLICY_2025_03_24_critic_target_model_{NODE_NUM}_{MS_NUM}_{AIMS_NUM}.pth"

    try:
        if os.path.exists(actor_url) and os.path.exists(critic_url):
            agent.actor_target.load_state_dict(torch.load(actor_url, map_location=torch.device('cpu')))
            agent.critic_target.load_state_dict(torch.load(critic_url, map_location=torch.device('cpu')))
            # 网络复制
            agent.actor.load_state_dict(agent.actor_target.state_dict())
            agent.critic.load_state_dict(agent.critic_target.state_dict())
            agent.actor.eval()
            agent.critic.eval()
            print("模型成功加载")
        else:
            print("模型文件不存在，请检查路径")
    except Exception as e:
        print(f"加载模型时出错: {e}")

if __name__ == '__main__':
    agent = Agent_on_policy.PPOAgent(state_dim, action_dim)
    load_model(agent)
    state = init_state()
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
        state = next_state.copy()
    print(episode_reward)
    target_deploy = get_deploy_for_new_state(state)
    target_rout = get_each_request_rout(target_deploy)
    total_delay = cal_total_delay(target_deploy,target_rout)
    print("微服务部署方案：")
    print(target_deploy)
    print("微服务路由方案：")
    for u in users:
        print(f"请求到达率{u.lamda}", end=' ')
        print(' ')
        print("服务请求：", end=' ')
        for i in requests.get(u):
            print(i.id, end=' ')
        print(' ')
        for i in marker.get(u):
            print(i, end=' ')
        print(' ')
        print(f"用户{u.id}的路由转发表")
        for i in range(len(target_rout[u.id])):
            print(f"处理微服务{requests.get(u)[i].id}的服务器元组{target_rout[u.id][i]}")
    print(f"资源剩余情况:{get_resource_for_new_state(state)}")
    print(f"网络总延迟:{total_delay}")
