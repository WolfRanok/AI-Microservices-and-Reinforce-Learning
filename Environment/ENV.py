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
    # 每一个服务器的已用资源和剩余资源的情况，资源有：CPU、GPU、内存
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
    """
    从状态中获取部署情况
    :param state:
    :return:
    """
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

def get_rout(state):
    rout = []
    for i in range(NODE_NUM):
        rout_node = state[(MS_NUM + AIMS_NUM) * NODE_NUM + i * (MS_NUM + AIMS_NUM) * NODE_NUM
                          :(MS_NUM + AIMS_NUM) * NODE_NUM + (i + 1) * (MS_NUM + AIMS_NUM) * NODE_NUM]
        rout_node = np.reshape(rout_node, ((MS_NUM + AIMS_NUM), NODE_NUM))
        rout.append(rout_node)
    return rout


def get_ms_image(ms, aims, users, requests, marke):
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
    :param users:
    :param node_list:
    :return:
    '''
    node = []
    for i in range(USER_NUM):
        user = users[i]
        node_idx = 0
        dis = float('inf')
        for item in node_list:
            dis = min(cal_dis_user_node(user, item), dis)
            if dis == cal_dis_user_node(user, item):
                node_idx = item.id
        node.append(node_idx)
    return node


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
    for ms in range(MS_NUM):
        ms_request = []
        for item in requests:
            item
        for node in range(NODE_NUM):
            ms = []

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
    resource = get_resource(state)
    # r = get_rout(state)
    print("状态：\n",state)
    print("状态中的部署方案:\n", d)
    print("状态中的资源情况:\n", resource)
    # print(r)

    ms = ms_initial()
    aims = aims_initial()
    user = user_initial()
    node_list = edge_initial()
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
    print(ms_image)

"""
输出实例：
状态：
 [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.  21.  21.   0.   0.   2.   4.   0.   0. 313. 371.]
状态中的部署方案:
 [[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]]
状态中的资源情况:
 [  0.   0.  21.  21.   0.   0.   2.   4.   0.   0. 313. 371.]
用户集： [<Environment.ENV_DEF.USER object at 0x0000025D98E7B110>, <Environment.ENV_DEF.USER object at 0x0000025D9A19AA20>, <Environment.ENV_DEF.USER object at 0x0000025D9A1988F0>]
请求集： {<Environment.ENV_DEF.USER object at 0x0000025D98E7B110>: [<Environment.ENV_DEF.MS object at 0x0000025D9A19AAE0>, <Environment.ENV_DEF.MS object at 0x0000025D9A19AB10>, <Environment.ENV_DEF.AIMS object at 0x0000025D9A19ACF0>, <Environment.ENV_DEF.AIMS object at 0x0000025D9A19AC30>], <Environment.ENV_DEF.USER object at 0x0000025D9A19AA20>: [<Environment.ENV_DEF.MS object at 0x0000025D9A19A960>, <Environment.ENV_DEF.MS object at 0x0000025D9A19AAB0>, <Environment.ENV_DEF.MS object at 0x0000025D9A19AD20>], <Environment.ENV_DEF.USER object at 0x0000025D9A1988F0>: [<Environment.ENV_DEF.MS object at 0x0000025D9A19ADB0>, <Environment.ENV_DEF.MS object at 0x0000025D9A19A9C0>]}
标记集合 {<Environment.ENV_DEF.USER object at 0x0000025D98E7B110>: [0, 0, 1, 1], <Environment.ENV_DEF.USER object at 0x0000025D9A19AA20>: [0, 0, 0], <Environment.ENV_DEF.USER object at 0x0000025D9A1988F0>: [0, 0]}
用户 0 的位置： 43.770227854494614 43.063758740878164
用户 1 的位置： 39.798255789234496 83.78376441951345
用户 2 的位置： 51.51932372339469 80.1496591292033
服务器 0 的位置： 28.617622359950026 56.5341515714332
服务器 1 的位置： 11.129328279505911 73.57647855696638
用户 0 到服务器 0 之间的距离： 20.27444046780999
用户 0 到服务器 1 之间的距离： 44.6817009036093
用户 1 到服务器 0 之间的距离： 29.454167179709792
用户 1 到服务器 1 之间的距离： 30.431826912589813
用户 2 到服务器 0 之间的距离： 32.896506236247944
用户 2 到服务器 1 之间的距离： 40.92136892618072
[0, 0, 0]
基础微服务的处理速率： [3, 2, 4, 2, 3]
AI微服务的处理速率： [1.2574850299401201, 0.9813084112149534, 1.47887323943662]
服务请求的到达率： [3, 6, 5]
0 3 0 2
0 1 4
0 4
[5. 3. 0. 2. 4. 3. 0. 3.]
"""
