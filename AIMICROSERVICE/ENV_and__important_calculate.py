import numpy as np

from AIMICROSERVICE.ENV_DEF import *

# 1326

Ms_Tolerate_Time = 5
Aims_Tolerate_Time = 10
v = 300000  # 波的速度
c = 300000  # 光的速度
np.random.seed(1256)
random.seed(1236)
global is_f
is_f = 0
all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, bandwidth, data, request_data = environment_initialization()
connected_lines, graph = connect_nodes_within_range(node_list, initial_range=10)


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
def get_deploy(state):
    deploy = state[0:MA_AIMS_NUM * NODE_NUM]
    deploy = np.reshape(deploy, (MA_AIMS_NUM, NODE_NUM))
    return deploy
def get_resource(state):
    """
    从状态中获取资源分配情况
    :param state: state
    :return:
    """
    resource = state[MA_AIMS_NUM * NODE_NUM:MA_AIMS_NUM * NODE_NUM + RESOURCE * 2 * NODE_NUM]
    return resource
def get_rout(state):
    start = (MA_AIMS_NUM + RESOURCE * 2) * NODE_NUM
    forward = state[start:start + USER_NUM * (NODE_NUM + 1) * NODE_NUM * MA_AIMS_NUM]
    forward = np.reshape(forward, (USER_NUM, NODE_NUM + 1, NODE_NUM, MA_AIMS_NUM))
    return forward
def get_ms_image_from_state(state):
    start = MA_AIMS_NUM * NODE_NUM + 3 * 2 * NODE_NUM + USER_NUM * (NODE_NUM * NODE_NUM + NODE_NUM)
    return state[start:start + USER_NUM * MA_AIMS_NUM]


def get_ms_dependency():
    dependency = np.zeros((MA_AIMS_NUM, MA_AIMS_NUM))
    for user in users:
        request = requests.get(user)
        pre_ms = request[0].id
        for ms_item in request[1:]:
            if isinstance(ms_item, MS):
                dependency[pre_ms][ms_item.id] = 1
                pre_ms = ms_item.id
            else:
                dependency[pre_ms][ms_item.id + MS_NUM] = 1
                pre_ms = ms_item.id + MS_NUM
    return dependency
def rout_to_forward(rout, forward):
    for request_idx in range(len(rout)):
        for row in rout[request_idx]:
            for ele in row:
                forward[request_idx][ele[0] + 1][ele[1]][ele[2]] = ele[3]
    return forward

def get_ms_deploy_order():
    idx_of_ms = []
    for user in users:
        request = requests.get(user)
        for ms in request:
            if isinstance(ms, MS):
                idx = user.id * MA_AIMS_NUM + ms.id
            else:
                idx = user.id * MA_AIMS_NUM + ms.id + MS_NUM
            idx_of_ms.append(idx)
    # for idx in idx_of_ms:
    #     print(ms_image[int((idx-idx%18)/18)][idx%18])
    return idx_of_ms

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

def get_ms_image():
    ms_image = np.zeros(MA_AIMS_NUM)
    ms_lamda = np.zeros(MA_AIMS_NUM)
    # request_lamda = get_user_lamda(users)
    # print(request_lamda)
    for user in users:
        lamda = user.lamda
        request = requests.get(user)
        single_marke = marker.get(user)
        for item1, item2 in zip(request, single_marke):
            if item2 == 0:
                ms_lamda[item1.id] += lamda
            else:
                ms_lamda[MS_NUM + item1.id] += lamda
    for i in range(MA_AIMS_NUM):
        rho = ms_lamda[i] / all_ms_alpha[i]
        ms_image[i] += math.ceil(rho)
    return ms_image


def get_each_req_ms_image():
    ms_image = np.zeros((USER_NUM, MA_AIMS_NUM))
    for user in users:
        lamda = user.lamda
        request = requests.get(user)
        single_marke = marker.get(user)
        for item1, item2 in zip(request, single_marke):
            if item2 == 0:
                ms_image[user.id][item1.id] = math.ceil(lamda / all_ms_alpha[item1.id]) + 1
            else:
                ms_image[user.id][item1.id + MS_NUM] = math.ceil(lamda / all_ms_alpha[item1.id + MS_NUM]) + 1
    return ms_image


def optimize_rout_node(ms_node_dict, request):
    '''
    去除不可达的节点
    :param ms_node_dict:
    :param request:
    :param node_graph:
    :return:
    '''
    result = ms_node_dict
    first_ms = request[0]
    list_node = []  # 后续进行递归检查需要用到，每一行保存这相应微服务的部署服务器

    for ms_item in request[1:]:
        node1 = ms_node_dict.get(first_ms)
        list_node.append(node1)
        node2 = ms_node_dict.get(ms_item)
        lag = {i: 0 for i in node2}
        if len(node1) == 0:
            new_node = []
            list_node.append(new_node)
            result[ms_item] = new_node

        else:
            new_node = node2.copy()
            for i in node1:
                for j in node2:
                    # print("服务器id",i.id,j.id)
                    if graph[i.id][j.id] != 0:
                        # print("yes")
                        lag[j] += 1
            # print(lag)
            for item in lag:
                if lag.get(item) == 0:
                    new_node.remove(item)
            # print(new_node==node2)
            list_node.append(new_node)
            result[ms_item] = new_node
        first_ms = ms_item
    sur_ms = request[-1]
    for ms_item in request[::-1]:
        if ms_item == sur_ms:
            continue
        node1 = ms_node_dict.get(sur_ms)
        list_node.append(node1)
        node2 = ms_node_dict.get(ms_item)
        lag = {i: 0 for i in node2}
        new_node = node2.copy()
        if len(node1) != 0:
            for i in node1:
                for j in node2:
                    # print("服务器id",i.id,j.id)
                    if graph[i.id][j.id] != 0:
                        # print("yes")
                        lag[j] += 1
            # print(lag)
            for item in lag:
                if lag.get(item) == 0:
                    new_node.remove(item)
            # print(new_node==node2)
            list_node.append(new_node)
            result[ms_item] = new_node
        sur_ms = ms_item
    return result


def cal_probability(node2, ms, ms_node_list, deploy):
    '''
    计算node1转发到node2上的转发概率
    :param node2:
    :param ms:
    :param node_list: 部署了ms的服务器集合
    :param node_bandwidth:
    :param deploy: 服务部署方案
    :return:
    '''
    total_ma_image = 0
    total_bandwidth = 0
    for item in ms_node_list:
        total_bandwidth += bandwidth[item.id]
        total_ma_image += deploy[ms.id][item.id]
    p = (bandwidth[node2.id] + deploy[ms.id][node2.id]) / (total_bandwidth + total_ma_image)
    return ms, node2, p


def get_each_request_rout(deploy):
    '''
    根据部署方案和服务请求生成每一条服务请求的处理路径图，每一个节点表示服务器，每一个边表示转发概率
    第一个节点是固定的，通过get_first_node（）函数获得
    :param deploy:
    :param users:
    :param requests:
    :return: 返回服务请求路由路径集合，每一条服务请求的路由路径图用邻接表存储。
    邻接表中用元组来表示路由转发(上一个节点，当前需要处理的微服务，当前节点，接收概率)
    '''
    all_user_rout = []
    for user in users:
        request = requests.get(user)
        idx = 0
        # node_list[node_idx] = EDGE_NODE(first_node)
        # 生成当前服务请求中各个微服务所在的节点集合
        # ms_node_dict: ms1:[node1, node2],ms2:[...]
        ms_node_dict = {}
        for ms_item in request:
            this_ms_node = []
            if marker.get(user)[idx] == 0:
                current_node = deploy[ms_item.id]
            else:
                current_node = deploy[ms_item.id + MS_NUM]
            for node_idx in range(NODE_NUM):
                if current_node[node_idx] != 0:
                    this_ms_node.append(node_list[node_idx])
                    # this_ms_node.append(EDGE_NODE(node_idx))
            ms_node_dict[ms_item] = this_ms_node
            idx += 1
        ms_node_dict = optimize_rout_node(ms_node_dict, request)
        all_node_list = []  # 存储了路由转发图中所有出现的节点，节点id会有重复
        for item in ms_node_dict:
            some_node = ms_node_dict.get(item)
            for node in some_node:
                all_node_list.append(node)
        # print(all_node_list)
        this_user_rout_path_p = []
        # 第一个微服务的转发情况需要特殊处理
        pre_node_list = ms_node_dict.get(request[0]).copy()
        first_ms_rout = []
        total_acc_delay = 0
        acc_delay = np.zeros(len(ms_node_dict.get(request[0])))
        for node, idx in zip(ms_node_dict.get(request[0]), range(len(ms_node_dict.get(request[0])))):
            acc_delay[idx] = cal_dis_user_node(user, node)
            total_acc_delay += acc_delay[idx]
        for first_node, idx in zip(ms_node_dict.get(request[0]), range(len(ms_node_dict.get(request[0])))):
            first_ms_rout.append((-1, first_node.id, request[0].id, acc_delay[idx] / total_acc_delay))
        this_user_rout_path_p.append(first_ms_rout)
        for item in request[1:]:
            all_node_of_this_ms = ms_node_dict.get(item).copy()
            ms_rout = []
            for node in pre_node_list:
                new_all_node_of_this_ms = []
                for node1 in all_node_of_this_ms:
                    if graph[node.id][node1.id] != 0:
                        new_all_node_of_this_ms.append(node1)
                for node2 in new_all_node_of_this_ms:
                    _, _, p = cal_probability(node2, item, new_all_node_of_this_ms, deploy)
                    rout = (node.id, node2.id, item.id, p)
                    ms_rout.append(rout)
                if not ms_rout:
                    continue
            pre_node_list = all_node_of_this_ms
            this_user_rout_path_p.append(ms_rout)
        # print(this_user_rout_path_p)
        all_user_rout.append(this_user_rout_path_p)
    return all_user_rout


def jiechen(n):
    k = 1
    if n == 0:
        return 1
    else:
        for i in range(1, n + 1):
            k *= i
        return k


def cal_ms_delay(deploy, lamda, ms, node):
    '''
    :param ms_deploy:
    :param a:
    :return:
    '''
    global is_f
    if isinstance(ms, MS):
        ms_proc_delay = Ms_Tolerate_Time
        alpha = all_ms_alpha[ms.id]
        if deploy[ms.id][node.id] == 0:
            return ms_proc_delay
        num = int(deploy[ms.id][node.id])
    else:
        ms_proc_delay = Aims_Tolerate_Time
        alpha = all_ms_alpha[ms.id + AIMS_NUM]
        if deploy[ms.id + AIMS_NUM][node.id] == 0:
            return ms_proc_delay
        num = int(deploy[ms.id + AIMS_NUM][node.id])
    rh0 = lamda / alpha
    rh1 = lamda / (num * alpha)
    if rh1 > 0.0 and rh1 < 0.99:
        v1 = 0
        for n in range(num):
            v2 = jiechen(n)
            v3 = math.pow(rh0, n) / v2
            v1 += v3
        v5 = jiechen(num)
        p0 = math.pow((v1 + math.pow(rh0, num) / (v5 * (1 - rh1))), -1)
        ms_proc_delay = 1 / alpha + rh1 * math.pow(rh0, num) * p0 / (lamda * v5 * math.pow((1 - rh1), 2))
    if ms_proc_delay >= Ms_Tolerate_Time or ms_proc_delay >= Aims_Tolerate_Time:
        is_f += 1
    return ms_proc_delay


def cal_total_delay(deploy, all_user_rout):
    '''
    网络时延的组成：发送时延和接收时延，等待时延，传输时延，传送时延
    :param users: 用户集
    :param node_list: 节点集
    :param all_ms_proc_delay: 微服务在服务器上的等待延迟
    :param all_user_rout: 请求路由表
    :param bandwidth: 带宽
    :param ms_data: 数据大小
    :return:
    '''
    # print(deploy)
    global is_f
    total_access_delay_each_user = np.zeros(USER_NUM)
    total_ms_proc_delay_each_user = np.zeros(USER_NUM)
    total_communication_delay_each_user = np.zeros(USER_NUM)
    total_reception_delay_each_user = np.zeros(USER_NUM)
    user_total_delay = np.zeros(USER_NUM)

    # ms_node_lamda = get_ms_node_lamda(deploy,all_user_rout)
    # 计算网络中所有微服务实例的处理延迟：

    # 计算发送延时，处理延迟、通信时延和接收时延
    for user in users:
        request = requests.get(user)
        lamda = user.lamda
        rout = all_user_rout[user.id]
        lamda_of_node = {}
        for idx in range(len(rout)):
            if idx == 0:
                # 此时只需要请求发送延迟和微服务处理延迟
                if not rout[idx]:
                    # 计算接入延迟
                    total_access_delay_each_user[user.id] += 2
                    # 计算当前微服务的处理延迟
                    if isinstance(request[idx], MS):
                        total_ms_proc_delay_each_user[user.id] += Ms_Tolerate_Time
                        is_f += 1
                    else:
                        total_ms_proc_delay_each_user[user.id] += Aims_Tolerate_Time
                        is_f += 1
                else:
                    # l = 0
                    for ele1 in rout[idx]:
                        node = node_list[ele1[1]]
                        ms = request[idx]
                        p = ele1[3]
                        ms_lamda = lamda * p
                        # 计算接入延迟(传输与传送)
                        if isinstance(ms,MS):
                            acc_delay = (data[ms.id]/bandwidth[node.id] + cal_dis_user_node(user, node)/v) * p
                        else:
                            acc_delay = (data[MS_NUM+ms.id]/bandwidth[node.id] + cal_dis_user_node(user, node)/v) * p
                        total_access_delay_each_user[user.id] += acc_delay
                        # 计算当前微服务的处理延迟
                        proc_delay1 = cal_ms_delay(deploy, ms_lamda, ms, node) * p
                        total_ms_proc_delay_each_user[user.id] += proc_delay1
                        # 保存本次流量分流情况
                        lamda_of_node[node.id] = ms_lamda
                        # l += ms_lamda
                    # print(lamda_of_node, l)
            else:
                if not rout[idx]:
                    if isinstance(request[idx], MS):
                        total_ms_proc_delay_each_user[user.id] += Ms_Tolerate_Time
                        total_communication_delay_each_user[user.id] += Ms_Tolerate_Time
                        is_f += 1
                    else:
                        total_ms_proc_delay_each_user[user.id] += Aims_Tolerate_Time
                        total_communication_delay_each_user[user.id] += Aims_Tolerate_Time
                        is_f += 1
                else:
                    new_lamda_of_node = {}
                    for ele2 in rout[idx]:
                        pre_node = node_list[ele2[0]]
                        node = node_list[ele2[1]]
                        ms = request[idx]
                        p = ele2[3]
                        ms_lamda = lamda_of_node[pre_node.id] * p
                        if node.id in new_lamda_of_node:
                            new_lamda_of_node[node.id] += ms_lamda
                        else:
                            new_lamda_of_node[node.id] = ms_lamda
                        # 计算处理延迟
                        proc_delay2 = cal_ms_delay(deploy, ms_lamda, ms, node) * p
                        total_ms_proc_delay_each_user[user.id] += proc_delay2
                        # 计算通信延迟
                        if isinstance(ms, MS):
                            this_data = data[ms.id]
                        else:
                            this_data = data[ms.id + MS_NUM]
                        trans_delay = this_data / bandwidth[node.id]
                        prop_delay = cal_dis(pre_node, node) / c
                        if prop_delay != 0:
                            total_communication_delay_each_user[user.id] += (trans_delay + prop_delay) * p
                    lamda_of_node = new_lamda_of_node
        # 服务请求的发送时延需要单独计算(传输与传送)
        if not rout[len(rout) - 1]:
            # F+=1
            total_reception_delay_each_user[user.id] += 2
        else:
            for key, value in lamda_of_node.items():
                node = node_list[key]
                p = value / lamda
                # print(f"user:{user.id},node:{node.id},p:{p}")
                ms = requests.get(user)[-1]
                if isinstance(ms,MS):
                    ms_idx = ms.id
                else:
                    ms_idx = ms.id+MS_NUM
                rec_delay = data[ms_idx]/bandwidth[node.id]+cal_dis_user_node(user, node) / v
                total_reception_delay_each_user[user.id] += rec_delay * p
        user_total_delay[user.id] = total_access_delay_each_user[user.id] + total_ms_proc_delay_each_user[user.id] \
                                    + total_communication_delay_each_user[user.id] + total_reception_delay_each_user[
                                        user.id]

    # print(f"服务请求接入时延：{total_access_delay_each_user}，总和为：{total_access_delay_each_user.sum()}")
    # print(f"服务请求处理时延：{total_ms_proc_delay_each_user}，总和为：{total_ms_proc_delay_each_user.sum()}")
    # print(f"服务请求通信时延：{total_communication_delay_each_user}，总和为：{total_communication_delay_each_user.sum()}")
    # print(f"服务请求发送时延：{total_reception_delay_each_user}，总和为：{total_reception_delay_each_user.sum()}")
    # print(f"服务请求响应时延：{user_total_delay}，总和为：{user_total_delay.sum()}")

    total_delay = user_total_delay.sum()
    return total_delay


def cal_each_D_max():
    '''
    网络时延的组成：发送时延和接收时延，等待时延，传输时延，传送时延
    :param users: 用户集
    :param node_list: 节点集
    :param all_ms_proc_delay: 微服务在服务器上的等待延迟
    :param all_user_rout: 请求路由表
    :param bandwidth: 带宽
    :param ms_data: 数据大小
    :return:
    '''
    # print(deploy)
    user_max_delay = np.zeros(USER_NUM)
    # ms_node_lamda = get_ms_node_lamda(deploy,all_user_rout)
    # 计算网络中所有微服务实例的处理延迟：

    # 计算发送延时，处理延迟、通信时延和接收时延
    for user in users:
        request = requests.get(user)
        user_max_delay[user.id] += 2
        count = 0
        for ms in request:
            count += 1
            if isinstance(ms, MS):
                user_max_delay[user.id] += Ms_Tolerate_Time
                if count < len(request):
                    user_max_delay[user.id] += Ms_Tolerate_Time
            else:
                user_max_delay[user.id] += Aims_Tolerate_Time
                if count < len(request):
                    user_max_delay[user.id] += Aims_Tolerate_Time
        user_max_delay[user.id] += 2
    return user_max_delay, user_max_delay.sum()


def cal_each_service_delay(deploy, all_user_rout):
    '''
    网络时延的组成：发送时延和接收时延，等待时延，传输时延，传送时延
    :param users: 用户集
    :param node_list: 节点集
    :param all_ms_proc_delay: 微服务在服务器上的等待延迟
    :param all_user_rout: 请求路由表
    :param bandwidth: 带宽
    :param ms_data: 数据大小
    :return:
    '''
    # print(deploy)
    global is_f
    total_access_delay_each_user = np.zeros(USER_NUM)
    total_ms_proc_delay_each_user = np.zeros(USER_NUM)
    total_communication_delay_each_user = np.zeros(USER_NUM)
    total_reception_delay_each_user = np.zeros(USER_NUM)
    user_total_delay = np.zeros(USER_NUM)

    # ms_node_lamda = get_ms_node_lamda(deploy,all_user_rout)
    # 计算网络中所有微服务实例的处理延迟：

    # 计算发送延时，处理延迟、通信时延和接收时延
    for user in users:
        request = requests.get(user)
        lamda = user.lamda
        rout = all_user_rout[user.id]
        lamda_of_node = {}
        for idx in range(len(rout)):
            if idx == 0:
                # 此时只需要请求发送延迟和微服务处理延迟
                if not rout[idx]:
                    # 计算接入延迟
                    total_access_delay_each_user[user.id] += 2
                    # 计算当前微服务的处理延迟
                    if isinstance(request[idx], MS):
                        total_ms_proc_delay_each_user[user.id] += Ms_Tolerate_Time
                        is_f += 1
                    else:
                        total_ms_proc_delay_each_user[user.id] += Aims_Tolerate_Time
                        is_f += 1
                else:
                    # l = 0
                    for ele1 in rout[idx]:
                        node = node_list[ele1[1]]
                        ms = request[idx]
                        p = ele1[3]
                        ms_lamda = lamda * p
                        # 计算接入延迟
                        acc_delay = (cal_dis_user_node(user, node) / v) * p
                        total_access_delay_each_user[user.id] += acc_delay
                        # 计算当前微服务的处理延迟
                        proc_delay1 = cal_ms_delay(deploy, ms_lamda, ms, node) * p
                        total_ms_proc_delay_each_user[user.id] += proc_delay1
                        # 保存本次流量分流情况
                        lamda_of_node[node.id] = ms_lamda
                        # l += ms_lamda
                    # print(lamda_of_node, l)
            else:
                if not rout[idx]:
                    if isinstance(request[idx], MS):
                        total_ms_proc_delay_each_user[user.id] += Ms_Tolerate_Time
                        total_communication_delay_each_user[user.id] += Ms_Tolerate_Time
                        is_f += 1
                    else:
                        total_ms_proc_delay_each_user[user.id] += Aims_Tolerate_Time
                        total_communication_delay_each_user[user.id] += Aims_Tolerate_Time
                        is_f += 1
                else:
                    new_lamda_of_node = {}
                    for ele2 in rout[idx]:
                        pre_node = node_list[ele2[0]]
                        node = node_list[ele2[1]]
                        ms = request[idx]
                        p = ele2[3]
                        ms_lamda = lamda_of_node[pre_node.id] * p
                        if node.id in new_lamda_of_node:
                            new_lamda_of_node[node.id] += ms_lamda
                        else:
                            new_lamda_of_node[node.id] = ms_lamda
                        # 计算处理延迟
                        proc_delay2 = cal_ms_delay(deploy, ms_lamda, ms, node) * p
                        total_ms_proc_delay_each_user[user.id] += proc_delay2
                        # 计算通信延迟
                        if isinstance(ms, MS):
                            this_data = data[ms.id]
                        else:
                            this_data = data[ms.id + MS_NUM]
                        trans_delay = this_data / bandwidth[node.id]
                        prop_delay = cal_dis(pre_node, node) / c
                        if prop_delay != 0:
                            total_communication_delay_each_user[user.id] += (trans_delay + prop_delay) * p
                    lamda_of_node = new_lamda_of_node
        # 服务请求的发送时延需要单独计算
        if not rout[len(rout) - 1]:
            # F+=1
            total_reception_delay_each_user[user.id] += 2
        else:
            for key, value in lamda_of_node.items():
                node = node_list[key]
                p = value / lamda
                # print(f"user:{user.id},node:{node.id},p:{p}")
                rec_delay = cal_dis_user_node(user, node) / v
                total_reception_delay_each_user[user.id] += rec_delay * p
        user_total_delay[user.id] = total_access_delay_each_user[user.id] + total_ms_proc_delay_each_user[user.id] \
                                    + total_communication_delay_each_user[user.id] + total_reception_delay_each_user[
                                        user.id]
    # print(f"服务请求接入时延：{total_access_delay_each_user}，总和为：{total_access_delay_each_user.sum()}")
    # print(f"服务请求处理时延：{total_ms_proc_delay_each_user}，总和为：{total_ms_proc_delay_each_user.sum()}")
    # print(f"服务请求通信时延：{total_communication_delay_each_user}，总和为：{total_communication_delay_each_user.sum()}")
    # print(f"服务请求发送时延：{total_reception_delay_each_user}，总和为：{total_reception_delay_each_user.sum()}")
    # print(f"服务请求响应时延：{user_total_delay}，总和为：{user_total_delay.sum()}")
    return user_total_delay


def cal_load_balance(state):
    resource = get_resource(state)
    cpu_occ = resource[0:NODE_NUM]
    total_cpu = resource[0:NODE_NUM] + resource[NODE_NUM:NODE_NUM * 2]
    gpu_occ = resource[NODE_NUM * 2:NODE_NUM * 3]
    total_gpu = resource[NODE_NUM * 2:NODE_NUM * 3] + resource[NODE_NUM * 3:NODE_NUM * 4]
    mem_occ = resource[NODE_NUM * 4:NODE_NUM * 5]
    total_mem = resource[NODE_NUM * 4:NODE_NUM * 5] + resource[NODE_NUM * 5:NODE_NUM * 6]

    u_cpu = cpu_occ / total_cpu
    u_gpu = np.zeros((NODE_NUM))
    for r in range(NODE_NUM):
        if total_gpu[r] != 0:
            u_gpu[r] = gpu_occ[r] / total_gpu[r]
    u_mem = mem_occ / total_mem

    avg_cpu = np.full(u_cpu.size, np.mean(u_cpu))
    avg_gpu = np.full(u_gpu.size, np.mean(u_gpu))
    avg_mem = np.full(u_mem.size, np.mean(u_mem))

    # print(u_cpu+u_gpu+u_mem)
    v_cpu = np.mean((u_cpu - avg_cpu) ** 2)
    v_gpu = np.mean((u_gpu - avg_gpu) ** 2)
    v_mem = np.mean((u_mem - avg_mem) ** 2)

    return v_cpu + v_gpu + v_mem


def show_graph():
    '''
    输出用户与服务器之间的位置关系图
    '''
    x_list = []
    y_list = []
    m = []
    for i in node_list:
        x, y = i.get_location()
        x_list.append(x)
        y_list.append(y)
        m.append(i.id)
    plt.scatter(x_list, y_list, c='red', marker='*')
    x_mean = sum(x_list) / len(x_list)
    y_mean = sum(y_list) / len(y_list)
    plt.scatter(x_mean, y_mean, c='green', marker='*')
    for xi, yi, mi in zip(x_list, y_list, m):
        # 给x和y添加偏移量，别和点重合在一起了
        plt.text(xi, yi, mi)
    # 节点连接
    connected_lines, V = connect_nodes_within_range(node_list, initial_range=10)
    for (node1, node2) in connected_lines:
        node1, node2 = node_list[node1], node_list[node2]
        plt.plot([node1.x, node2.x], [node1.y, node2.y], c='red', linestyle='-', linewidth=0.5)
    # 画出用户
    user_x_list = []
    user_y_list = []
    for i in users:
        x, y = i.get_location()
        user_x_list.append(x)
        user_y_list.append(y)
    for xi, yi, mi in zip(user_x_list, user_y_list, m):
        # 给x和y添加偏移量，别和点重合在一起了
        plt.text(xi, yi, mi)
    plt.scatter(user_x_list, user_y_list, c='blue')
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # show_graph()
    state = initial_state()
    print(data)
    print(request_data)
    print(state)
    ms_image = get_ms_image()  # all_ms_alpha, users, requests, marker
    print(f"微服务实例数量{ms_image}")
    cpu = 0
    gpu = 0
    mem = 0
    for idx in range(MA_AIMS_NUM):
        ms = all_ms[idx]
        cpu += ms.get_cpu() * ms_image[idx]
        gpu += ms.get_gpu() * ms_image[idx]
        mem += ms.get_memory() * ms_image[idx]
    print(f"所需cpu,gpu,mem资源{cpu, gpu, mem}")
    node_cpu = 0
    node_gpu = 0
    node_mem = 0
    for item in node_list:
        node_cpu += item.cpu
        node_gpu += item.gpu
        node_mem += item.memory
        if item.gpu != 0:
            print(item.id)
    print(f"拥有的cpu,gpu,mem资源{node_cpu, node_gpu, node_mem}")
    each = get_each_req_ms_image()
    print(f"微服务实例数量：{each}")
    print(f"请求需要的微服务实例数量：{np.sum(each, -1)}")
    # deploy = [[0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 1, 0, 9, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    #           [0, 0, 0, 0, 18, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 2, 2, 4, 0, 1, 0, 0, 0, 0]]
    # deploy = [[0, 0, 2, 2, 1, 2, 0, 0, 0, 1],
    #           [0, 0, 1, 0, 2, 2, 1, 0, 0, 0],
    #           [0, 0, 0, 0, 1, 0, 2, 0, 2, 0],
    #           [2, 2, 2, 1, 2, 1, 2, 0, 1, 0],
    #           [0, 1, 0, 0, 0, 2, 1, 0, 0, 1],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 2, 0, 0, 1, 1, 0, 0, 1, 1]]
    # deploy = [[3, 0, 0, 3, 0, 1, 0, 1, 0, 2],
    #           [0, 1, 1, 0, 2, 1, 3, 1, 1, 0],
    #           [0, 1, 1, 0, 0, 2, 1, 3, 0, 0],
    #           [1, 1, 2, 4, 0, 2, 2, 3, 1, 2],
    #           [0, 0, 2, 0, 2, 1, 1, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 2, 1, 1, 2, 0, 0, 0, 2, 1]]
    deploy = [[0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
              [0, 0, 2, 1, 1, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 1, 1, 2, 0, 1],
              [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 0, 1, 1, 2, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 2, 0, 0],
              [0, 0, 0, 0, 0, 2, 3, 2, 0, 0],
              [0, 0, 0, 0, 0, 3, 2, 2, 0, 0]]
    rout = get_each_request_rout(deploy)
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
        for i in range(len(rout[u.id])):
            print(f"处理微服务{requests.get(u)[i].id}的服务器元组{rout[u.id][i]}")
    print(cal_total_delay(deploy, rout))
    print(is_f)
    # ms_node_lamda=get_ms_node_lamda(deploy,rout)
    # print("请求到达率",ms_node_lamda)
    for u in users:
        print(f"用户{u.id}的到达率为{u.lamda}")
    delay = cal_total_delay(deploy, rout)
    print(delay)
    # forward = get_rout(state)
    # # print(f"转发表{forward}")
    # rout_to_forward(rout, forward)
    # print(f"转发表{get_rout(state)[0][5][7][7]}")
    # loadb = cal_load_balance(state)
    # print(loadb)
    # print(get_ms_deploy_order())
    # print(get_ms_dependency())
