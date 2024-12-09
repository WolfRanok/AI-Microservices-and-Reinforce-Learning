import numpy as np
from Environment.ENV_DEF import *

random.seed(123)


def initial_state():
    '''
    deploy_state:(MS_NUM+AIMS_NUM)*NODE_NUM
    rout:NODE_NUM*(MS_NUM+AIMS_NUM)*NODE_NUM
    :return:
    '''
    deploy_state = np.zeros(shape=(MS_NUM + AIMS_NUM, NODE_NUM))
    # rout_state = np.zeros(shape=(NODE_NUM, MS_NUM + AIMS_NUM, NODE_NUM))
    CPU = np.zeros(shape=(2, NODE_NUM))
    GPU = np.zeros(shape=(2, NODE_NUM))
    Memory = np.zeros(shape=(2, NODE_NUM))
    node_list = []
    for i in range(NODE_NUM):
        edge_node = EDGE_NODE(i)
        node_list.append(edge_node)
        CPU[1][i] = edge_node.cpu  # 初始化剩余cpu资源
        GPU[1][i] = edge_node.gpu  # 初始化剩余gpu资源
        Memory[1][i] = edge_node.memory  # 初始化剩余memory资源
    deploy_state = np.reshape(deploy_state, (1, (MS_NUM + AIMS_NUM) * NODE_NUM))
    # rout_state = np.reshape(rout_state, (1, NODE_NUM * (MS_NUM + AIMS_NUM) * NODE_NUM))
    CPU = np.reshape(CPU, (1, 2 * NODE_NUM))
    GPU = np.reshape(GPU, (1, 2 * NODE_NUM))
    Memory = np.reshape(Memory, (1, 2 * NODE_NUM))
    resource = np.append(CPU, GPU)
    resource = np.append(resource, Memory)
    # state = np.append(deploy_state, rout_state)
    # state = np.append(state, resource)
    state = np.append(deploy_state, resource)
    return state


def get_deploy(state):
    deploy = state[0:(MS_NUM + AIMS_NUM) * NODE_NUM]
    deploy = np.reshape(deploy, ((MS_NUM + AIMS_NUM), NODE_NUM))
    return deploy


def get_resource(state):
    """
    从状态中获取资源分配情况
    :param state: state
    :return:
    """
    resource = state[(MS_NUM + AIMS_NUM) * NODE_NUM:]

    return resource


# def get_rout(state):
#     rout = []
#     for i in range(NODE_NUM):
#         rout_node = state[(MS_NUM+AIMS_NUM)*NODE_NUM+i*(MS_NUM+AIMS_NUM)*NODE_NUM
#                           :(MS_NUM+AIMS_NUM)*NODE_NUM+(i+1)*(MS_NUM+AIMS_NUM)*NODE_NUM]
#         rout_node = np.reshape(rout_node,((MS_NUM+AIMS_NUM), NODE_NUM))
#         rout.append(rout_node)
#     return rout

def get_ms_image(ms, aims, users, requests, marke):
    """
    根据给定的条件计算镜像分配情况
    :param ms:
    :param aims:
    :param users:
    :param requests:
    :param marke:
    :return:
    """
    ms_image = np.zeros(MS_NUM + AIMS_NUM)
    ms_lamda = np.zeros(MS_NUM + AIMS_NUM)
    # request_lamda = get_user_lamda(users)
    # print(request_lamda)
    for user in users:
        lamda = user.lamda
        request = requests.get(user)
        single_marke = marke.get(user)
        for item1, item2 in zip(request, single_marke):
            if item2 == 0:
                ms_lamda[item1.id] += lamda
            else:
                ms_lamda[MS_NUM + item1.id] += lamda
    alpha_list = np.append(get_ms_alpha(ms), get_aims_alpha(aims))
    for i in range(MS_NUM + AIMS_NUM):
        rho = ms_lamda[i] / alpha_list[i]
        ms_image[i] += math.ceil(rho)
    return ms_image


def get_first_node(users, node_list):
    '''
    获得服务请求接收节点集
    :param users: 用户对象，用于提供用户地址
    :param node_list:服务节点列表，每一个服务器节点都有一个地址
    :return: 最近的服务器节点
    '''
    node = []
    for i in range(len(users)):
        user = users[i]
        node_idx = 0
        dis = float('inf')
        for item in node_list:
            dis = min(cal_dis_user_node(user, item), dis)
            if dis == cal_dis_user_node(user, item):
                node_idx = item.id
        node.append(node_idx)
    return node


def generation(node_graph, node, nodelist):
    result = {}
    i = 0
    for idx in nodelist:
        if idx != 0 & node_graph[i][node] != 0:
            result[node].append(idx)


def optimize_rout_node(ms_node_dict, request, node_graph):
    '''
    去除不可达的节点
    :param ms_node_dict:
    :param request:
    :param node_graph:
    :return:
    '''
    result = ms_node_dict
    first_ms = request[0]
    for ms_item in request[1:]:
        node1 = ms_node_dict.get(first_ms)
        node2 = ms_node_dict.get(ms_item)
        lag = {i: 0 for i in node2}
        if len(node1) == 0:
            new_node = []
            result[ms_item] = new_node
        else:
            new_node = node2
            for i in node1:
                for j in node2:
                    if node_graph[i.id][j.id] != 0:
                        lag[j] += 1
            for item in lag:
                if lag.get(item) == 0:
                    new_node.remove(item)
            result[ms_item] = new_node
        first_ms = ms_item
    return result


def cal_probability(node2, ms, node_list, node_bandwidth, deploy):
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
    for item in node_list:
        total_bandwidth += node_bandwidth[item.id]
        total_ma_image += deploy[ms.id][item.id]
    p = (node_bandwidth[node2.id] + deploy[ms.id][node2.id]) / (total_bandwidth + total_ma_image)
    return ms, node2, p


def get_each_request_rout(node_graph, deploy, bandwidth, nodes, users, requests, marke):
    '''
    根据部署方案和服务请求生成每一条服务请求的处理路径图，每一个节点表示服务器，每一个边表示转发概率
    第一个节点是固定的，通过get_first_node（）函数获得
    :param deploy:
    :param users:
    :param requests:
    :return: 返回服务请求路由路径集合，每一条服务请求的路由路径图用邻接表存储
    '''
    first_node_list = get_first_node(users, nodes)
    all_user_rout = []
    for user in users:
        request = requests.get(user)
        first_node = first_node_list[user.id]
        idx = 1
        node_idx = 0
        # node_list[node_idx] = EDGE_NODE(first_node)
        # 生成当前服务请求中各个微服务所在的节点集合
        # ms_node_dict: ms1:[node1, node2],ms2:[...]
        ms_node_dict = {}
        this_ms_node = []
        this_ms_node.append(EDGE_NODE(first_node))
        ms_node_dict[request[0]] = this_ms_node
        for ms_item in request[1:]:
            this_ms_node = []
            if marke.get(user)[idx] == 0:
                current_node = deploy[ms_item.id]
            else:
                current_node = deploy[ms_item.id + MS_NUM]
            for node_idx in range(NODE_NUM):
                if current_node[node_idx] != 0:
                    this_ms_node.append(EDGE_NODE(node_idx))
            ms_node_dict[ms_item] = this_ms_node
            idx += 1
        # print(ms_node_dict)
        # print(user)
        # for item in ms_node_dict:
        #     node = ms_node_dict.get(item)
        #     print(node)
        print("字典1", ms_node_dict)
        ms_node_dict = optimize_rout_node(ms_node_dict, request, node_graph)
        print("字典2", ms_node_dict)
        all_node_list = []  # 存储了路由转发图中所有出现的节点，节点id会有重复
        for item in ms_node_dict:
            some_node = ms_node_dict.get(item)
            for node in some_node:
                all_node_list.append(node)
        # print(all_node_list)
        this_user_rout_path_p = []
        # 第一个微服务的转发情况需要特殊处理
        first_ms_rout = []
        first_ms_rout.append((request[0].id, first_node, 1))
        this_user_rout_path_p.append(first_ms_rout)
        # print("rout",this_user_rout_path_p)
        for item in request[1:]:
            all_node_of_this_ms = ms_node_dict.get(item)
            ms_rout = []
            for node in all_node_of_this_ms:
                _, _, p = cal_probability(node, item, all_node_of_this_ms, bandwidth, deploy)
                rout = (item.id, node.id, p)
                ms_rout.append(rout)
            this_user_rout_path_p.append(ms_rout)
        # print(this_user_rout_path_p)
        all_user_rout.append(this_user_rout_path_p)
    return all_user_rout


def get_ms_node_lamda(state, users, requests, node_list):
    '''

    :param state: 一维向量
    :param users: list
    :param requests: dict
    :param node_list: list
    :return:
    '''
    ms_node_lamda = []
    first_node = get_first_node(users, node_list)
    deploy = get_deploy(state)
    # rout = get_rout(state)
    # 微服务在服务器上的流量分为两个部分：直接流量和间接流量
    for ms in range(MS_NUM):
        for node in range(NODE_NUM):
            # 计算直接流量
            lamda1 = 0
            for item in users:
                request = requests.get(item)
                if first_node[item.id] == node & request[0].id == ms & deploy[ms][node] != 0:
                    lamda1 += item.lamda
            # 计算间接流量

    return ms_node_lamda


def jiechen(n):
    k = 1
    if n == 0:
        return 1
    else:
        for i in range(1, n + 1):
            k *= i
        return k


def cal_ms_delay(ms_deploy):
    '''
    :param ms_deploy: (NODE_NUM*MS_NUM)
    :param a:
    :return:
    '''
    ms_delay = []
    for i in range(NODE_NUM):
        ms_on_node_delay = []
        for j in range(MS_NUM):
            num = ms_deploy[i][j]


if __name__ == '__main__':
    state = initial_state()
    d = get_deploy(state)
    print(state)
    print(d)

    ms = ms_initial()
    aims = aims_initial()
    user = user_initial()
    node_list = edge_initial()
    print(node_list)
    users, user_list, marke = get_user_request(user)
    print("用户集：", users)
    print("请求集：", user_list)
    print("标记集合", marke)
    for item in users:
        print("用户", item.id, "的位置：", item.x, item.y)
    for item in node_list:
        print("服务器", item.id, "的位置：", item.x, item.y)
    for item1 in users:
        for item2 in node_list:
            print("用户", item1.id, "到服务器", item2.id, "之间的距离：", cal_dis_user_node(item1, item2))
    print(get_first_node(users, node_list))
    msalpha = get_ms_alpha(ms)
    aimsalpha = get_aims_alpha(aims)
    servicelamda = get_user_lamda(user)
    print("基础微服务的处理速率：", msalpha)
    print("AI微服务的处理速率：", aimsalpha)
    print("服务请求的到达率：", servicelamda)
    for item in users:
        for i in user_list.get(item):
            print(i.id, end=' ')
        print(' ')
    ms_image = get_ms_image(ms, aims, users, user_list, marke)

    _, graph = connect_nodes_within_range(node_list, initial_range=10)
    deploy = []  # 这里随机生成用作测试
    random.seed(123)
    for _ in range(MS_NUM + AIMS_NUM):
        deploy_ms = np.random.randint(low=0, high=2, size=NODE_NUM)
        deploy.append(deploy_ms)
    print(deploy)
    rout = get_each_request_rout(graph, deploy, get_bandwidth_with_node(), node_list, users, user_list, marke)
    print(rout)
