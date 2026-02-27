import math

import numpy as np

from Environment import *
import torch
import Environmental_parameters.arrival_rate_of_service.arrival_rate_4_6


if __name__ == '__main__':
    torch.cuda.set_device(0)

    env_args = Arguments.get_args()
    # env_args = Environmental_parameters.arrival_rate_of_service.arrival_rate_4_6.get_args()
    env = ENV(env_args)
    ms_list = np.ones(env.MS_AIMS_NUM)
    node_b= np.zeros(env.NODE_NUM)
    for n in env.node_list:
        node_b[n.id] = math.pow(1+math.exp(-n.bandwidth/10),-1)
    print(node_b)
    for ms_item in env.Ms_types_instances:
        if isinstance(ms_item,MS):
            ms_list[ms_item.id] += 1
        else:
            ms_list[ms_item.id+env.MS_NUM] += 1
    cpu = np.zeros(env.MS_AIMS_NUM)
    gpu = np.zeros(env.MS_AIMS_NUM)
    mem = np.zeros(env.MS_AIMS_NUM)
    for idx in range(env.MS_AIMS_NUM):
        ms = env.all_ms_list[idx]
        cpu[idx] += ms.get_cpu() * ms_list[idx]
        gpu[idx] += ms.get_gpu() * ms_list[idx]
        mem[idx] += ms.get_memory() * ms_list[idx]
    print(f"所需cpu资源{cpu}")
    print(f"所需gpu资源{gpu}")
    print(f"所需mem资源{mem}")
    print(f"所需cpu,gpu,mem资源{cpu.sum(), gpu.sum(), mem.sum()}")
    node_cpu = np.zeros(env.NODE_NUM)
    node_gpu = np.zeros(env.NODE_NUM)
    node_mem = np.zeros(env.NODE_NUM)
    for item in env.node_list:
        node_cpu[item.id] += item.cpu
        node_gpu[item.id] += item.gpu
        node_mem[item.id] += item.memory
        if item.gpu != 0:
            print(item.id)
    print(f"服务器拥有的cpu资源{node_cpu}")
    print(f"服务器拥有的gpu资源{node_gpu}")
    print(f"服务器拥有的mem资源{node_mem}")
    print(f"服务器拥有的cpu,gpu,mem资源{node_cpu.sum(), node_gpu.sum(), node_mem.sum()}")

    # deployment = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    # [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    deployment = [[0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
                  [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    env.MLP_state(False)
    env.GNN_state(False)
    rout = env.get_each_request_rout()
    for user in env.user_list:
        print(f"用户{user.id}的路由转发表")
        for i in range(len(rout.get(user))):
            print(f"处理微服务{user.request_chain[i].id}的服务器元组{rout.get(user)[i]}")
    delay = env.get_total_delay()
    print(delay)



