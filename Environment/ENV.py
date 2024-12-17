import numpy as np
from Environment.ENV_DEF import *
# 1326
random.seed(1326)
np.random.seed(25)
Ms_Tolerate_Time = 3
Aims_Tolerate_Time = 5
v = 100 # 波的速度
c = 10000 # 光的速度

all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, \
        bandwidth, data = environment_initialization()
connected_lines, graph = connect_nodes_within_range(node_list, initial_range=10)
def initial_state():
    '''
    deploy_state:MA_AIMS_NUM*NODE_NUM
    rout:NODE_NUM*MA_AIMS_NUM*NODE_NUM
    :return:
    '''
    deploy_state = np.zeros(shape=(MA_AIMS_NUM, NODE_NUM))
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
    deploy_state = np.reshape(deploy_state, (1, MA_AIMS_NUM * NODE_NUM))
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
    deploy = state[0:MA_AIMS_NUM*NODE_NUM]
    deploy = np.reshape(deploy, (MA_AIMS_NUM, NODE_NUM))
    return deploy

def get_resource(state):
    """
    从状态中获取资源分配情况
    :param state: state
    :return:
    """
    resource = state[MA_AIMS_NUM * NODE_NUM:]

    return resource

# def get_rout(state):
#     rout = []
#     for i in range(NODE_NUM):
#         rout_node = state[MA_AIMS_NUM*NODE_NUM+i*MA_AIMS_NUM*NODE_NUM
#                           :MA_AIMS_NUM*NODE_NUM+(i+1)*MA_AIMS_NUM*NODE_NUM]
#         rout_node = np.reshape(rout_node,(MA_AIMS_NUM, NODE_NUM))
#         rout.append(rout_node)
#     return rout

def updata_state(state, action, ms_idx, all_ms):
    state_new = np.reshape(state, (MA_AIMS_NUM + 2 * RESOURCE, NODE_NUM))
    action = np.reshape(action, (1, NODE_NUM))
    # CY: 确定性策略更新，每次选择value最大的行动
    act_idx = np.argmax(action)
    state_new[ms_idx][act_idx] += 1
    state_new[MA_AIMS_NUM][act_idx] += all_ms[ms_idx].cpu
    state_new[MA_AIMS_NUM + 1][act_idx] -= all_ms[ms_idx].cpu
    if ms_idx>=MS_NUM:
        state_new[MS_NUM + AIMS_NUM + 2][act_idx] += all_ms[ms_idx].gpu
        state_new[MS_NUM + AIMS_NUM + 3][act_idx] -= all_ms[ms_idx].gpu
    state_new[MA_AIMS_NUM + 4][act_idx] += all_ms[ms_idx].memory
    state_new[MA_AIMS_NUM + 5][act_idx] -= all_ms[ms_idx].memory
    state_new = np.reshape(state_new, (1, (MA_AIMS_NUM + 2 * RESOURCE) * NODE_NUM))
    return state_new, act_idx

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
                ms_lamda[MS_NUM+item1.id] += lamda
    for i in range(MA_AIMS_NUM):
        rho = ms_lamda[i]/all_ms_alpha[i]
        ms_image[i] += math.ceil(rho)
    return ms_image

def get_first_node():
    '''
    获得服务请求接收节点集，拿到的是服务器的编号，但需要用到服务器的具体属性时，需要配合node_list使用
    :param users:
    :param node_list:
    :return:
    '''
    node = []
    for i in range(len(users)):
        user = users[i]
        node_idx = 0
        dis = float('inf')
        for item in node_list:
            dis = min(cal_dis_user_node(user, item),dis)
            if dis==cal_dis_user_node(user, item):
                node_idx = item.id
        node.append(node_idx)
    return node

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
    node_list = [] # 后续进行递归检查需要用到，每一行保存这相应微服务的部署服务器
    # for idx in range(len(request)-1):
    #     ms_idx = idx+1
    #     if ms_idx!=len(request)-1:
    #         pre_node = ms_node_dict.get(request[ms_idx-1])
    #         this_node = ms_node_dict.get(request[ms_idx])
    #         suc_node = ms_node_dict.get(request[ms_idx+1])
    #         pre_lag = {i: 0 for i in this_node}
    #         suc_lag = {i: 0 for i in this_node}
    #         if not pre_node:
    #             new_node = []
    #             node_list.append(new_node)
    #             result[all_ms[ms_idx]] = new_node
    #         else:
    #             new_node = this_node.copy()
    #             for i, j in zip(pre_node,suc_node):
    #                 for k in this_node:
    #                     # print("服务器id",i.id,j.id)
    #                     if graph[i.id][k.id] != 0:
    #                         # print("yes")
    #                         pre_lag[k] += 1
    #                     if graph[j.id][k.id] != 0:
    #                         # print("yes")
    #                         suc_lag[k] += 1
    #             for item1, item2 in zip(pre_lag,suc_lag):
    #                 if pre_lag.get(item1)==0 or suc_lag.get(item2)==0:
    #                     new_node.remove(item1)
    #             node_list.append(new_node)
    #             result[all_ms[ms_idx]] = new_node
    #     else:
    #         pre_node = ms_node_dict.get(request[ms_idx - 1])
    #         this_node = ms_node_dict.get(request[ms_idx])
    #         pre_lag = {i: 0 for i in this_node}
    #         if not pre_node:
    #             new_node = []
    #             node_list.append(new_node)
    #             result[all_ms[ms_idx]] = new_node
    #         else:
    #             new_node = this_node.copy()
    #             for i in pre_node:
    #                 for k in this_node:
    #                     # print("服务器id",i.id,j.id)
    #                     if graph[i.id][k.id] != 0:
    #                         # print("yes")
    #                         pre_lag[k] += 1
    #             for item1 in pre_lag:
    #                 if pre_lag.get(item1)==0:
    #                     new_node.remove(item1)
    #             node_list.append(new_node)
    #             result[all_ms[ms_idx]] = new_node
    for ms_item in request[1:]:
        node1 = ms_node_dict.get(first_ms)
        node_list.append(node1)
        node2 = ms_node_dict.get(ms_item)
        lag = {i: 0 for i in node2}
        if len(node1)==0:
            new_node = []
            node_list.append(new_node)
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
            node_list.append(new_node)
            result[ms_item] = new_node
        first_ms = ms_item
    for ms_item in request[::-1]:
        node1 = ms_node_dict.get(first_ms)
        node_list.append(node1)
        node2 = ms_node_dict.get(ms_item)
        lag = {i: 0 for i in node2}
        if len(node1)==0:
            new_node = []
            node_list.append(new_node)
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
            node_list.append(new_node)
            result[ms_item] = new_node
        first_ms = ms_item
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
    p = (bandwidth[node2.id]+deploy[ms.id][node2.id])/(total_bandwidth+total_ma_image)
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
    first_node_list = get_first_node()
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
        this_ms_node.append(node_list[first_node])
        # this_ms_node.append(EDGE_NODE(first_node))
        ms_node_dict[request[0]] = this_ms_node
        for ms_item in request[1:]:
            this_ms_node = []
            if marker.get(user)[idx] == 0:
                current_node = deploy[ms_item.id]
            else:
                current_node = deploy[ms_item.id+MS_NUM]
            for node_idx in range(NODE_NUM):
                if current_node[node_idx]!=0:
                    this_ms_node.append(node_list[node_idx])
                    # this_ms_node.append(EDGE_NODE(node_idx))
            ms_node_dict[ms_item] = this_ms_node
            idx += 1
        # print(ms_node_dict)
        # print(user)
        # for item in ms_node_dict:
        #     node = ms_node_dict.get(item)
        #     print(node)
        # print("字典1",ms_node_dict)
        ms_node_dict = optimize_rout_node(ms_node_dict, request)
        # print("字典2",ms_node_dict)
        all_node_list = [] # 存储了路由转发图中所有出现的节点，节点id会有重复
        for item in ms_node_dict:
            some_node = ms_node_dict.get(item)
            for node in some_node:
                all_node_list.append(node)
        # print(all_node_list)
        this_user_rout_path_p = []
        # 第一个微服务的转发情况需要特殊处理
        first_ms_rout= []
        first_ms_rout.append((-1,first_node,request[0].id,1))
        this_user_rout_path_p.append(first_ms_rout)
        # print("rout",this_user_rout_path_p)
        pre_node_list = ms_node_dict.get(request[0]).copy()
        for item in request[1:]:
            all_node_of_this_ms = ms_node_dict.get(item).copy()
            # print("fuwuqi",all_node_of_this_ms)
            # print(len(all_node_of_this_ms))
            ms_rout = []
            for node in pre_node_list:
                # new_all_node_of_this_ms = all_node_of_this_ms.copy()
                this_pre_node_ms_rout = []
                new_all_node_of_this_ms = []
                for node1 in all_node_of_this_ms:
                    if graph[node.id][node1.id]!=0:
                        new_all_node_of_this_ms.append(node1)
                for node2 in new_all_node_of_this_ms:
                    _, _, p = cal_probability(node2, item, new_all_node_of_this_ms, deploy)
                    rout = (node.id, node2.id, item.id, p)
                    this_pre_node_ms_rout.append(rout)
                if not this_pre_node_ms_rout:
                    continue
                ms_rout.append(this_pre_node_ms_rout)
            pre_node_list = all_node_of_this_ms
            this_user_rout_path_p.append(ms_rout)
        # print(this_user_rout_path_p)
        all_user_rout.append(this_user_rout_path_p)
    return all_user_rout

def get_ms_node_lamda(deploy,all_user_rout):
    '''
    :param state: 一维向量
    :param users: list
    :param requests: dict
    :param node_list: list
    :return: 服务器上每个微服务的到达率
    '''
    ms_node_lamda= np.zeros(shape=(MA_AIMS_NUM,NODE_NUM))
    first_node = get_first_node()
    # rout = get_rout(state)
    # 微服务在服务器上的流量分为两个部分：直接流量和间接流量
    # 计算直接流量
    for item in users:
        request = requests.get(item)
        if deploy[request[0].id][first_node[item.id]] != 0:
            ms_node_lamda[request[0].id][first_node[item.id]] += item.lamda
    for idx1 in range(len(all_user_rout)):
        user = users[idx1]
        request = requests.get(user)
        rout = all_user_rout[idx1]
        lamda = user.lamda
        node_lamda_list = []
        for idx2 in range(len(rout)):# 单个请求链的每一层
            if idx2==0:
                node = rout[idx2][0][1]
                ms = rout[idx2][0][2]
                p = rout[idx2][0][3]
                # print(node,ms,p)
                this_lamda = lamda*p
                ms_node_lamda[ms][node] += this_lamda
                node_lamda_list.append((node,this_lamda))
                # print(node_lamda_list)
            else:# 第二层往后
                new_node_lamda_list = [] # 记录新一层中服务器上的微服务到达率情况
                for i in range(len(rout[idx2])):
                    single_of_rout = rout[idx2][i]
                    # print("zhenggeyuanzu",single_of_rout)
                    for idx3 in range(len(single_of_rout)):
                        tup = single_of_rout[idx3]
                        # print("yuanzu",tup)
                        node = tup[1]
                        ms = tup[2]
                        p = tup[3]
                        this_lamda = 0
                        for item1 in node_lamda_list:
                            if item1[0]==tup[0]:
                                this_lamda = item1[1] * p
                        # 微服务类型不同，但是编号可能一样
                        if isinstance(request[idx2], MS):
                            ms_node_lamda[ms][node] += this_lamda
                        else:
                            ms_node_lamda[ms + MS_NUM][node] += this_lamda
                        # 更新 new_node_lamda_list
                        if not new_node_lamda_list:
                            new_node_lamda_list.append((node, this_lamda))
                        else:
                            flag = 0
                            for item2 in new_node_lamda_list:
                                if item2[0] == node:
                                    flag += 1
                                    new_lamda = item2[1] + this_lamda
                                    new_node_lamda_list.remove(item2)
                                    new_item = (node, new_lamda)
                                    new_node_lamda_list.append(new_item)
                                    break
                            if flag == 0:
                                new_node_lamda_list.append((node, this_lamda))
                            # new_node_lamda_list = new_node_lamda_list
                node_lamda_list = new_node_lamda_list
            # print("服务请求",idx1,"的第",idx2,"层各个节点的到达率情况",node_lamda_list)
    return ms_node_lamda

def jiechen(n):
    k = 1
    if n == 0:
        return 1
    else:
        for i in range(1, n + 1):
            k *= i
        return k

# def cal_ms_delay(deploy, all_user_rout):
#     '''
#     :param ms_deploy:
#     :param a:
#     :return:
#     '''
#     ms_node_lamda = get_ms_node_lamda(deploy,all_user_rout)
#     ms_proc_delay = np.zeros(shape=(MA_AIMS_NUM,NODE_NUM))
#     for i in range(MA_AIMS_NUM):
#         alpha = all_ms_alpha[i]
#         for j in range(NODE_NUM):
#             if deploy[i][j]==0:
#                 continue
#             lamda = ms_node_lamda[i][j]
#             num = int(deploy[i][j])
#             rh0 = lamda/alpha
#             rh1 = lamda/(num*alpha)
#             if rh1>0 and rh1<1:
#                 v1 = 0
#                 for n in range(num):
#                     v2 = jiechen(n)
#                     v3 = math.pow(rh0, n) / v2
#                     v1 += v3
#                 v5 = jiechen(num)
#                 p0 = math.pow((v1 + math.pow(rh0, num) / (v5 * (1 - rh1))), -1)
#                 ms_proc_delay[i][j] = 1 / alpha + rh1 * math.pow(rh0, num) * p0 / (lamda * v5 * math.pow((1 - rh1), 2))
#             elif rh1>=1:
#                 if i < MS_NUM:
#                     ms_proc_delay[i][j] = Ms_Tolerate_Time
#                 else:
#                     ms_proc_delay[i][j] = Aims_Tolerate_Time
#     return ms_proc_delay
def cal_ms_delay(deploy,ms_node_lamda, ms, node):
    '''
    :param ms_deploy:
    :param a:
    :return:
    '''
    if isinstance(ms,MS):
        ms_proc_delay = Ms_Tolerate_Time
        alpha = all_ms_alpha[ms.id]
        lamda = ms_node_lamda[ms.id][node.id]
        if deploy[ms.id][node.id] == 0:
            return ms_proc_delay
        num = int(deploy[ms.id][node.id])
    else:
        ms_proc_delay = Aims_Tolerate_Time
        alpha = all_ms_alpha[ms.id+AIMS_NUM]
        lamda = ms_node_lamda[ms.id+AIMS_NUM][node.id]
        if deploy[ms.id+AIMS_NUM][node.id] == 0:
            return ms_proc_delay
        num = int(deploy[ms.id+AIMS_NUM][node.id])
    rh0 = lamda / alpha
    rh1 = lamda / (num * alpha)
    if rh1 > 0 and rh1 < 1:
        v1 = 0
        for n in range(num):
            v2 = jiechen(n)
            v3 = math.pow(rh0, n) / v2
            v1 += v3
        v5 = jiechen(num)
        p0 = math.pow((v1 + math.pow(rh0, num) / (v5 * (1 - rh1))), -1)
        ms_proc_delay = 1 / alpha + rh1 * math.pow(rh0, num) * p0 / (lamda * v5 * math.pow((1 - rh1), 2))
    # if isinstance(ms, MS) and ms_proc_delay==Ms_Tolerate_Time:
    #     print("实例数不足")
    # elif isinstance(ms, AIMS) and ms_proc_delay==Aims_Tolerate_Time:
    #     print("实例数不足")
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
    total_ms_proc_delay = 0
    total_access_delay = 0
    total_reception_delay = 0
    total_communication_delay = 0
    ms_node_lamda = get_ms_node_lamda(deploy,all_user_rout)
    # 发送延时计算如下：
    first_node_list = get_first_node()
    for idx in range(len(users)):
        first_node = node_list[first_node_list[idx]]
        total_access_delay += cal_dis_user_node(users[idx],first_node)/v
    # 计算处理延迟、通信时延和接收时延
    for user in users:
        # print("user",user.id)
        request = requests.get(user)
        rout = all_user_rout[user.id]
        for idx in range(len(rout)):
            if idx==0:
                for item in rout[idx]:
                    node = node_list[item[1]]
                    ms = request[idx]
                    total_ms_proc_delay += cal_ms_delay(deploy, ms_node_lamda, ms, node)
                continue
            elif not rout[idx]:
                if isinstance(request[idx],MS):
                    total_ms_proc_delay += Ms_Tolerate_Time
                else:
                    total_ms_proc_delay += Aims_Tolerate_Time
            else:
                for row in rout[idx]:
                    # print(row)
                    for item1 in row:
                        ms = request[idx]
                        pre_node = node_list[item1[0]]
                        this_node = node_list[item1[1]]
                        if isinstance(ms, MS):
                            this_data = data[ms.id]
                        else:
                            this_data = data[ms.id + MS_NUM]
                        total_ms_proc_delay += cal_ms_delay(deploy, ms_node_lamda, ms, this_node)
                        trans_delay = this_data / bandwidth[this_node.id]
                        prop_delay = cal_dis(pre_node, this_node) / c
                        total_communication_delay += (trans_delay + prop_delay) * item1[3]
                        if idx == (len(rout)-1):
                            last_node = this_node
                            total_reception_delay += item1[3] * cal_dis_user_node(user, last_node) / v
    total_delay = total_ms_proc_delay+total_access_delay+total_reception_delay+total_communication_delay
    # print("等待时延",total_ms_proc_delay)
    # print("发送时延",total_access_delay)
    # print("接收时延",total_reception_delay)
    # print("通信时延",total_communication_delay)
    # print("总时延",total_delay)
    return total_delay



if __name__ == '__main__':
    # all_ms, all_ms_alpha, node_list, users, requests, service_lamda, marker, bandwidth, data = environment_initialization()
    # connected_lines, graph = connect_nodes_within_range(node_list, initial_range=10)
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
    for xi, yi, mi in zip(x_list, y_list, m):
        # 给x和y添加偏移量，别和点重合在一起了
        plt.text(xi, yi + 1, mi)
    user_x_list = []
    user_y_list = []
    for i in requests:
        x, y = i.get_location()
        user_x_list.append(x)
        user_y_list.append(y)
    # 节点连接
    print(connected_lines)
    for (node1, node2) in connected_lines:
        node1, node2 = node_list[node1], node_list[node2]
        plt.plot([node1.x, node2.x], [node1.y, node2.y], c='red', linestyle='-', linewidth=0.5)

    plt.scatter(user_x_list, user_y_list, c='blue')
    plt.tight_layout()
    plt.show()
    state = initial_state()
    print(state)
    # 初始化网络环境
    print("用户集：",users)
    print("请求集：",requests)
    print("标记集合", marker)

    for item in users:
        print("请求到达率",item.lamda)
        print("服务请求：")
        for i in requests.get(item):
            print(i.id, end=' ')
        print(' ')
    ms_image = get_ms_image() # all_ms_alpha, users, requests, marker
    print(ms_image)
    ms_idx = np.argmax(ms_image)
    print(ms_idx)

    print("graph",graph)
    # deploy = np.zeros(shape=(MA_AIMS_NUM,NODE_NUM))  # 这里随机生成用作测试
    deploy =[[1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
             [0,0,0,1,1,1,0,0,0,1],
             [0, 1, 0, 0,0,0,0, 2, 1, 0],
             [1,0,0,1,0,2,1,0,0,0],
             [0,2, 2,1,1,0, 0, 0, 0, 1],
             [0,0,0,0,0,0,0,0,0,0]]
    # np.random.seed(12)
    # for ms_idx in range(MA_AIMS_NUM):
    #     deploy[ms_idx] = np.random.randint(low=0, high=5, size=NODE_NUM)
    # print("服务部署方案",deploy)
    rout = get_each_request_rout(deploy)
    for i in range(len(users)):
        print("用户",i,"的路由转发表")
        for item in rout[i]:
            print(item)
    # print(rout[0])
    # print(rout[1])
    # print(rout[2])
    ms_node_lamda=get_ms_node_lamda(deploy,rout)
    print("请求到达率",ms_node_lamda)
    for u in users:
        print(u.lamda)
    total = 0
    for item in ms_node_lamda:
        total += sum(item)
    print(total)
    print(data)
    total_delay = cal_total_delay(deploy,rout)
    print(total_delay)
    print(get_ms_image())



