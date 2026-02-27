from ENV import *

def initial_state():
    '''
    deploy_state:MA_AIMS_NUM*NODE_NUM
    rout:NODE_NUM*MA_AIMS_NUM*NODE_NUM
    :return:
    '''
    deploy_state = np.zeros(shape=(MA_AIMS_NUM, NODE_NUM))
    # rout_state = np.zeros(shape=(USER_NUM, NODE_NUM + 1, NODE_NUM, MA_AIMS_NUM))
    CPU = np.zeros(shape=(2, NODE_NUM))
    GPU = np.zeros(shape=(2, NODE_NUM))
    Memory = np.zeros(shape=(2, NODE_NUM))
    for i in range(NODE_NUM):
        edge_node = node_list[i]
        CPU[1][i] = edge_node.cpu  # 初始化剩余cpu资源
        GPU[1][i] = edge_node.gpu  # 初始化剩余gpu资源
        Memory[1][i] = edge_node.memory  # 初始化剩余memory资源
    deploy_state = np.reshape(deploy_state, (1, MA_AIMS_NUM * NODE_NUM))
    # rout_state = np.reshape(rout_state, (1, USER_NUM * (NODE_NUM + 1) * NODE_NUM * MA_AIMS_NUM))
    CPU = np.reshape(CPU, (1, 2 * NODE_NUM))
    GPU = np.reshape(GPU, (1, 2 * NODE_NUM))
    Memory = np.reshape(Memory, (1, 2 * NODE_NUM))
    resource = np.append(CPU, GPU)
    resource = np.append(resource, Memory)

    state = np.append(deploy_state, resource)
    # state = np.append(state, rout_state)
    # 保存当前需要部署的微服务实例的ID，CPU,GPU,MEN资源
    MS_information = []
    m1 = requests.get(users[0])[0]
    MS_information.append(m1.id)
    MS_information.append(m1.get_cpu())
    MS_information.append(m1.get_gpu())
    MS_information.append(m1.get_memory())
    state = np.append(state, MS_information)
    # 保存正在部署微服务实例的用户请求信息，ID,到达率，位置
    Resquest_information = []
    user1 = users[0]
    Resquest_information.append(user1.id)
    Resquest_information.append(user1.get_lamda())
    user1_x, user1_y = user1.get_location()
    Resquest_information.append(user1_x)
    Resquest_information.append(user1_y)
    state = np.append(state, Resquest_information)
    ### 以下都是不会变化的值
    # 保存服务器信息，ID，位置
    node_information = []
    for node in node_list:
        id = node.id
        B = bandwidth[id]
        x = node.x
        y = node.y
        node_information.append(id)
        node_information.append(B)
        node_information.append(x)
        node_information.append(y)
    state = np.append(state, node_information)
    dependency = get_ms_dependency()
    dependency = np.reshape(dependency, (1, MA_AIMS_NUM * MA_AIMS_NUM))
    state = np.append(state, dependency)
    graph_1 = np.reshape(graph, (NODE_NUM * NODE_NUM))
    state = np.append(state, graph_1)
    return state

def updata_state(state, act_idx, ms_idx):
    state_new = state[:(MA_AIMS_NUM + 2 * RESOURCE) * NODE_NUM]
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
    state[:(MA_AIMS_NUM + 2 * RESOURCE) * NODE_NUM] = state_new.copy()
    return state