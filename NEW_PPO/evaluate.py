import os
import torch

import Arguments
from Environment import ENV
from Agent import *
from Normalization import *
from DQN import DQNAgent,evaluate_policy
from Object_Parameter_DEF import MS

dqn_url = 'Environmental_parameters/length_of_service/dqn_Model/dqn_target_net_length_of_service_4_6.pth'
def load_model(agent):
    """
    当模型存在时，用于加载模型
    :return: None
    """

    try:
        if os.path.exists(dqn_url):
            agent.q_net.load_state_dict(torch.load(dqn_url))
            agent.target_net.load_state_dict(torch.load(dqn_url))
            # # 网络复制
            # agent.actor.load_state_dict(agent.actor_target.state_dict())
            # agent.critic.load_state_dict(agent.critic_target.state_dict())
            print("模型成功加载")
        else:
            print("模型文件不存在，请检查路径")
    except Exception as e:
        print(f"加载模型时出错: {e}")

def text_evaluate(env, args):
    agent = DQNAgent(env, args)
    load_model(agent)
    evaluate_policy(env, agent, args)

    ms_traffic = env.get_microservices_traffic_distribution()
    sojourn_delay, _ = env.get_total_sojourn_delay(ms_traffic)
    communication_delay, _ = env.get_total_communication_delay()
    delay = env.get_total_delay()
    cpu = 0
    gpu = 0
    mem = 0
    for ms in env.all_ms_list:
        if isinstance(ms, MS):
            ms_idx = ms.id
        else:
            ms_idx = ms.id + env.MS_NUM
        for node in env.node_list:
            cpu += env.MS_instances_deployed_on_server[ms_idx][node.id] * ms.get_cpu()
            gpu += env.MS_instances_deployed_on_server[ms_idx][node.id] * ms.get_gpu()
            mem += env.MS_instances_deployed_on_server[ms_idx][node.id] * ms.get_memory()
    return delay, sojourn_delay, communication_delay, cpu, gpu, mem

if __name__ == '__main__':
    import Environmental_parameters.length_of_service.length_of_service_4_6

    args = Arguments.get_args()  # 获取训练参数
    env_args = Environmental_parameters.length_of_service.length_of_service_4_6.get_args()  # 获取环境参数

    env = ENV(env_args)

    delay, sojourn_delay, communication_delay, cpu, gpu, mem =  text_evaluate(env, args)

    print(f"total delay: {delay}")
    print(f"sojourn delay: {sojourn_delay}")
    print(f"communication delay: {communication_delay}")
    print(f"total cpu: {cpu}")
    print(f"total gpu: {gpu}")
    print(f"total mem: {mem}")