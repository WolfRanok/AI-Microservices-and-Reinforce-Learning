from ENV import *


def init_state():
    # 当前网络部署方案
    deploy_state = np.zeros(shape=(MA_AIMS_NUM, NODE_NUM))
    # 当前网络资源情况
    CPU = np.zeros(shape=(2, NODE_NUM))
    GPU = np.zeros(shape=(2, NODE_NUM))
    Memory = np.zeros(shape=(2, NODE_NUM))
    for i in range(NODE_NUM):
        edge_node = node_list[i]
        CPU[1][i] = edge_node.cpu  # 初始化剩余cpu资源
        GPU[1][i] = edge_node.gpu  # 初始化剩余gpu资源
        Memory[1][i] = edge_node.memory  # 初始化剩余memory资源
    # 将状态变为一维
    CPU = np.reshape(CPU, (1, 2 * NODE_NUM))
    GPU = np.reshape(GPU, (1, 2 * NODE_NUM))
    Memory = np.reshape(Memory, (1, 2 * NODE_NUM))
    RES = np.append(CPU, GPU)
    RES = np.append(RES, Memory)
    # 当前部署的服务请求基本信息
    current_request_info = []
    current_user = users[0]
    current_request = requests.get(current_user)  # 服务请求
    current_request_chain = get_request_chain(current_request)  # 服务请求链路
    current_request_arrive_rate = current_user.get_lamda()  # 服务请求到达率
    D_max = 10  # 服务请求最大容忍时延
    current_request_rout = get_this_request_rout(deploy_state, current_user, current_request)  # 服务请求的路由策略
    D = cal_this_service_delay(deploy_state, current_user, current_request_rout)  # 服务请求的响应时延
    current_request_forward = reshape_current_request_rout(current_request_rout)  # 路由方案形状转换，路由表变矩阵形式
    current_request_forward = np.reshape(current_request_forward, (-1))
    current_request_info = np.append(current_request_info, current_user.id)
    current_request_info = np.append(current_request_info, current_request_arrive_rate)
    current_request_info = np.append(current_request_info, D_max)
    current_request_info = np.append(current_request_info, D)
    current_request_info = np.append(current_request_info, current_request_chain)
    current_request_info = np.append(current_request_info, current_request_forward)
    # 当前部署的微服务实例的基本信息
    current_ms_info = []
    current_ms = current_request[0]  # 微服务实例
    if isinstance(current_ms, MS):  # 判断是微服务还是AI微服务
        current_ms_data = data[current_ms.id]  # data表示服务数据的大小
    else:
        current_ms_data = data[current_ms.id + MS_NUM]
    cpu_of_current_ms = current_ms.get_cpu()  # 微服务的cpu需求
    gpu_of_current_ms = current_ms.get_gpu()  # 微服务的gpu需求
    mem_of_current_ms = current_ms.get_memory()  # 微服务的mem需求
    current_ms_info = np.append(current_ms.id, current_ms_data)
    current_ms_info = np.append(current_ms_info, cpu_of_current_ms)
    current_ms_info = np.append(current_ms_info, gpu_of_current_ms)
    current_ms_info = np.append(current_ms_info, mem_of_current_ms)

    ### 以下都是不会变化的值
    # 保存服务器信息，ID，位置
    node_information = []
    for node in node_list:
        id = node.id
        B = bandwidth[id]
        node_information.append(id)
        node_information.append(B)
    dependency = get_ms_dependency()
    dependency = np.reshape(dependency, (MA_AIMS_NUM * MA_AIMS_NUM))
    graph_1 = np.reshape(graph, (NODE_NUM * NODE_NUM))

    state = np.append(current_ms_info, current_request_info)
    state = np.append(state, deploy_state)
    state = np.append(state, RES)
    state = np.append(state, node_information)
    state = np.append(state, dependency)
    state = np.append(state, graph_1)
    return state


def updata_state_for_new_state(state, act_idx, ms_idx):
    """
    进行资源分配
    :param state:
    :param act_idx:
    :param ms_idx:
    :return:
    """
    start = 9 + MA_AIMS_NUM + MA_AIMS_NUM * NODE_NUM
    state_new = state[start:start + (MA_AIMS_NUM + 2 * RESOURCE) * NODE_NUM]
    state_new = np.reshape(state_new, ((MA_AIMS_NUM + 2 * RESOURCE), NODE_NUM))
    # action = np.reshape(action, (1, NODE_NUM))
    # # CY: 确定性策略更新，每次选择value最大的行动
    # act_idx = np.argmax(action)
    state_new[ms_idx][act_idx] += 1
    state_new[MA_AIMS_NUM][act_idx] += all_ms[ms_idx].cpu
    state_new[MA_AIMS_NUM + 1][act_idx] -= all_ms[ms_idx].cpu
    if ms_idx >= MS_NUM:
        state_new[MA_AIMS_NUM + 2][act_idx] += all_ms[ms_idx].gpu
        state_new[MA_AIMS_NUM + 3][act_idx] -= all_ms[ms_idx].gpu
    state_new[MA_AIMS_NUM + 4][act_idx] += all_ms[ms_idx].memory
    state_new[MA_AIMS_NUM + 5][act_idx] -= all_ms[ms_idx].memory
    # state_new = np.reshape(state_new, (1, (MA_AIMS_NUM + 2 * RESOURCE) * NODE_NUM))
    state_new = np.ravel(state_new)
    state[start:start + (MA_AIMS_NUM + 2 * RESOURCE) * NODE_NUM] = state_new.copy()
    return state


def updata_rout_for_new_state(state, rout):
    forward = reshape_current_request_rout(rout)
    forward = np.reshape(forward, (-1))
    start = 9 + MA_AIMS_NUM
    state[start:start + MA_AIMS_NUM * NODE_NUM] = forward.copy()
    return state
