from Interaction import *

def init_node_res():
    node_free_res = np.zeros(shape=(3, NODE_NUM))
    for node_item in node_list:
        node_free_res[0][node_item.id] = node_item.get_cpu()
        node_free_res[1][node_item.id] = node_item.get_gpu()
        node_free_res[2][node_item.id] = node_item.get_memory()
    node_occ_res = np.zeros(shape=(3, NODE_NUM))
    return node_free_res, node_occ_res
def get_connect_nodes(node):
    result = []
    for node_item in node_list:
        if graph[node.id][node_item.id]!=0:
            result.append(node_item)
    return result
def get_each_ms_num():
    each_request_each_ms = get_each_req_ms_image()
    each_ms_num = np.zeros((MA_AIMS_NUM))
    for i in range(MA_AIMS_NUM):
        for j in range(USER_NUM):
            each_ms_num[i] += each_request_each_ms[j][i]
    return each_ms_num
def generate_state(current_ms, current_user, deploy, res_free, res_occ):
    # 当前网络资源情况
    CPU = np.zeros(shape=(2, NODE_NUM))
    GPU = np.zeros(shape=(2, NODE_NUM))
    Memory = np.zeros(shape=(2, NODE_NUM))
    for i in range(NODE_NUM):
        edge_node = node_list[i]
        CPU[0][i] = res_occ[0][edge_node.id]  # 初始化已占用cpu资源
        CPU[1][i] = res_free[0][edge_node.id]  # 初始化剩余cpu资源
        GPU[0][i] = res_occ[1][edge_node.id]  # 初始化已占用gpu资源
        GPU[1][i] = res_free[1][edge_node.id]  # 初始化剩余gpu资源
        Memory[0][i] = res_occ[2][edge_node.id]  # 初始化已占用memory资源
        Memory[1][i] = res_free[2][edge_node.id]  # 初始化剩余memory资源
    CPU = np.reshape(CPU, (-1))
    GPU = np.reshape(GPU, (-1))
    Memory = np.reshape(Memory, (-1))
    RES = np.append(CPU, GPU)
    RES = np.append(RES, Memory)
    # 当前部署的服务请求基本信息
    current_request = requests.get(current_user)  # 服务请求
    current_request_chain = get_request_chain(current_request)  # 服务请求链路
    current_request_arrive_rate = current_user.get_lamda()  # 服务请求到达率
    D_max = 10  # 服务请求最大容忍时延
    current_request_rout = get_this_request_rout(deploy, current_user, current_request)  # 服务请求的路由策略
    D = cal_this_service_delay(deploy, current_user, current_request_rout)  # 服务请求的响应时延
    current_request_forward = reshape_current_request_rout(current_request_rout)
    current_request_forward = np.reshape(current_request_forward, (-1))
    current_request_info = np.append(current_user.id, current_request_arrive_rate)
    current_request_info = np.append(current_request_info, D_max)
    current_request_info = np.append(current_request_info, D)
    current_request_info = np.append(current_request_info, current_request_chain)
    current_request_info = np.append(current_request_info, current_request_forward)
    # 当前部署的微服务实例的基本信息
    if isinstance(current_ms, MS):
        current_ms_data = data[current_ms.id]
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
    state = np.append(state, deploy)
    state = np.append(state, RES)
    state = np.append(state, node_information)
    state = np.append(state, dependency)
    state = np.append(state, graph_1)
    return state
def generate_reward(done, can_dep, state, current_ms, ms_mun_for_reward):
    deploy = get_deploy_for_new_state(state)
    rout = get_each_request_rout(deploy)
    T = cal_total_delay(deploy, rout)
    loadb = cal_load_balance(get_resource_for_new_state(state))
    if isinstance(current_ms, MS):
        if can_dep:
            ms_idx = current_ms.id
            r = 1 / ms_mun_for_reward[ms_idx]
        else:
            r = -1
    else:
        if can_dep:
            ms_idx = current_ms.id+MS_NUM
            r = 10 / ms_mun_for_reward[ms_idx]
        else:
            r = -5
    if done:
        r += 1 - 0.1 * T - 10*loadb
    return r

def get_random_deploy_dependency(transition_dict):
    random.seed()
    ms_dependency = get_ms_dependency()
    node_free_res, node_occ_res = init_node_res()
    deploy = np.zeros(shape=(MA_AIMS_NUM,NODE_NUM))
    ms_mun_for_reward = get_each_ms_num()
    done = False
    is_first = True
    for user_item in users:
        request = requests.get(user_item)
        arrival_rate = user_item.get_lamda()
        for ms_item in request:
            if isinstance(ms_item,MS):
                ms_num = math.ceil(arrival_rate/all_ms_alpha[ms_item.id])+1
                ms_id = ms_item.id
            else:
                ms_num = math.ceil(arrival_rate/all_ms_alpha[ms_item.id+MS_NUM])+1
                ms_id = ms_item.id+MS_NUM
            while ms_num!=0:
                candi_node = []
                depend_node = []
                connect_node = []
                for node_item in node_list:
                    if node_free_res[0][node_item.id] >= ms_item.get_cpu() and node_free_res[1][node_item.id] >= ms_item.get_gpu() and node_free_res[2][node_item.id] >= ms_item.get_memory():
                        candi_node.append(node_item)
                for node_item in candi_node:
                    for ms_idx in range(MA_AIMS_NUM):
                        if deploy[ms_idx][node_item.id] != 0 and ms_dependency[ms_item.id][ms_idx] != 0:
                            depend_node.append(node_item)
                # 如果具有依赖关系的服务器资源不足，就寻找这些服务器的相邻服务器
                depend_node_ = []
                if not depend_node:
                    for node_item in node_list:
                        for ms_idx in range(MA_AIMS_NUM):
                            if deploy[ms_idx][node_item.id] != 0 and ms_dependency[ms_item.id][ms_idx] != 0:
                                depend_node_.append(node_item)
                    for node_item in depend_node_:
                        for node_idx in range(NODE_NUM):
                            alternative_node = node_list[node_idx]
                            if graph[node_item.id][alternative_node.id]!=0 and (alternative_node in candi_node):
                                connect_node.append(alternative_node)
                can_deploy = True
                if depend_node:
                    node = random.choice(depend_node)
                elif connect_node:
                    node = random.choice(connect_node)
                elif candi_node:
                    node = random.choice(candi_node)
                else:
                    can_deploy = False
                # 记录部署前的网络状态
                state = generate_state(ms_item,user_item,deploy,node_free_res,node_occ_res)
                if is_first:
                    transition_dict['states'].append(state)
                    is_first=False
                else:
                    transition_dict['states'].append(state)
                    transition_dict['next_states'].append(state)
                action = node.id
                transition_dict['actions'].append(action)
                if can_deploy:
                    node_id = action
                    deploy[ms_id][node_id] += 1
                    node_free_res[0][node_id] -= ms_item.get_cpu()
                    node_free_res[1][node_id] -= ms_item.get_gpu()
                    node_free_res[2][node_id] -= ms_item.get_memory()
                    node_occ_res[0][node_id] += ms_item.get_cpu()
                    node_occ_res[1][node_id] += ms_item.get_gpu()
                    node_occ_res[2][node_id] += ms_item.get_memory()
                ms_num -= 1
                if ms_num==0 and user_item==users[-1] and ms_item==request[-1]:
                    # 表示最后一个微服务实例也部署完成了
                    state = generate_state(ms_item,user_item,deploy,node_free_res,node_occ_res)
                    transition_dict['next_states'].append(state)
                    done = True
                reward = generate_reward(done,can_deploy,state,ms_item,ms_mun_for_reward)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
    return deploy, node_free_res, node_occ_res
def get_random_deploy_connect(transition_dict):
    random.seed()
    ms_dependency = get_ms_dependency()
    node_free_res, node_occ_res = init_node_res()
    deploy = np.zeros(shape=(MA_AIMS_NUM,NODE_NUM))
    ms_mun_for_reward = get_each_ms_num()
    done = False
    is_first = True
    for user_item in users:
        request = requests.get(user_item)
        arrival_rate = user_item.get_lamda()
        pre_ms_nodes = []
        for ms_item in request:
            if isinstance(ms_item,MS):
                ms_num = math.ceil(arrival_rate/all_ms_alpha[ms_item.id])+1
                ms_id = ms_item.id
            else:
                ms_num = math.ceil(arrival_rate/all_ms_alpha[ms_item.id+MS_NUM])+1
                ms_id = ms_item.id+MS_NUM
            while ms_num!=0:
                can_deploy = True
                if ms_item==request[0]:
                    candi_node_for_frist = [] # 资源足够的服务器
                    for node_item in node_list:
                        if node_free_res[0][node_item.id] >= ms_item.get_cpu() and node_free_res[1][
                            node_item.id] >= ms_item.get_gpu() and node_free_res[2][
                            node_item.id] >= ms_item.get_memory():
                            candi_node_for_frist.append(node_item)
                    depend_node = []
                    for node_item in candi_node_for_frist:
                        for ms_idx in range(MA_AIMS_NUM):
                            if deploy[ms_idx][node_item.id] != 0 and ms_dependency[ms_item.id][ms_idx] != 0:
                                depend_node.append(node_item)
                    if depend_node:
                        node = random.choice(depend_node)
                        if node not in pre_ms_nodes:
                            pre_ms_nodes.append(node)
                    elif candi_node_for_frist:
                        node = random.choice(candi_node_for_frist)
                        if node not in pre_ms_nodes:
                            pre_ms_nodes.append(node)
                    else:
                        can_deploy = False
                        print("网络资源不足，部署失败")
                else:
                    this_ms_node = pre_ms_nodes.copy()
                    pre_ms_nodes.clear()
                    candi_node_for_then = []  # 资源足够的服务器
                    connect_node_or_this_node = []
                    for node_item in node_list:
                        if node_free_res[0][node_item.id] >= ms_item.get_cpu() and node_free_res[1][
                            node_item.id] >= ms_item.get_gpu() and node_free_res[2][
                            node_item.id] >= ms_item.get_memory():
                            candi_node_for_then.append(node_item)
                    for node_item in this_ms_node:
                        if node_item in candi_node_for_then:
                            connect_node_or_this_node.append(node_item)
                        connect_node = get_connect_nodes(node_item)
                        for adj_node in connect_node:
                            if adj_node in candi_node_for_then:
                                connect_node_or_this_node.append(adj_node)
                    if connect_node_or_this_node:
                        node = random.choice(connect_node_or_this_node)
                        if node not in pre_ms_nodes:
                            pre_ms_nodes.append(node)
                    elif candi_node_for_then:
                        node = random.choice(candi_node_for_then)
                        if node not in pre_ms_nodes:
                            pre_ms_nodes.append(node)
                    else:
                        can_deploy = False
                        print("网络资源不足，部署失败")
                # 记录部署前的网络状态
                state = generate_state(ms_item, user_item, deploy, node_free_res, node_occ_res)
                if is_first:
                    transition_dict['states'].append(state)
                    is_first = False
                else:
                    transition_dict['states'].append(state)
                    transition_dict['next_states'].append(state)
                action = node.id
                transition_dict['actions'].append(action)
                if can_deploy:
                    node_id = action
                    deploy[ms_id][node_id] += 1
                    node_free_res[0][node_id] -= ms_item.get_cpu()
                    node_free_res[1][node_id] -= ms_item.get_gpu()
                    node_free_res[2][node_id] -= ms_item.get_memory()
                    node_occ_res[0][node_id] += ms_item.get_cpu()
                    node_occ_res[1][node_id] += ms_item.get_gpu()
                    node_occ_res[2][node_id] += ms_item.get_memory()
                    ms_num -= 1
                if ms_num==0 and user_item==users[-1] and ms_item==request[-1]:
                    # 表示最后一个微服务实例也部署完成了
                    state = generate_state(ms_item,user_item,deploy,node_free_res,node_occ_res)
                    transition_dict['next_states'].append(state)
                    done = True
                reward = generate_reward(done,can_deploy,state,ms_item,ms_mun_for_reward)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
    return deploy, node_free_res, node_occ_res

if __name__ == '__main__':
    for _ in range(10):
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        deploy, node_free_res, node_occ_res = get_random_deploy_dependency(transition_dict)
        rout = get_each_request_rout(deploy)
        # print(deploy)
        # for u in users:
        #     print(f"请求到达率{u.lamda}", end=' ')
        #     print(' ')
        #     print("服务请求：", end=' ')
        #     for i in requests.get(u):
        #         print(i.id, end=' ')
        #     print(' ')
        #     for i in marker.get(u):
        #         print(i, end=' ')
        #     print(' ')
        #     print(f"用户{u.id}的路由转发表")
        #     for i in range(len(rout[u.id])):
        #         print(f"处理微服务{requests.get(u)[i].id}的服务器元组{rout[u.id][i]}")
        delay = cal_each_service_delay(deploy, rout)
        print(delay)
        print(sum(delay))
        res = np.zeros(shape=(6, NODE_NUM))
        res[0] = node_occ_res[0]
        res[1] = node_free_res[0]
        res[2] = node_occ_res[1]
        res[3] = node_free_res[1]
        res[4] = node_occ_res[2]
        res[5] = node_free_res[2]
        res = np.reshape(res, (-1))
        load = cal_load_balance(res)
        print(load)

