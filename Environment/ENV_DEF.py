import random
import math
import matplotlib.pyplot as plt
import numpy as np

MS_NUM = 4
AIMS_NUM = 3
NODE_NUM = 10
USER_NUM = 10
RESOURCE = 3
MA_AIMS_NUM = MS_NUM+AIMS_NUM

random.seed(123)
np.random.seed(123)
class MS:
    '''
    基础微服务拥有两种资源类型
    '''
    def __init__(self, id) -> None:
        self.id = id
        self.alpha = random.randint(2, 4)
        self.cpu = random.randint(1, 2)
        self.memory = random.randint(10, 15)

    def get_alpha(self):
        return self.alpha

    def get_cpu(self):
        return self.cpu

    def get_gpu(self):  # 普通微服务没有光gpu需求
        return 0

    def get_memory(self):
        return self.memory

class AIMS:
    '''
    AI微服务需要三种资源类型
    AI微服务的处理速率alpha由组成它的dnn网络的处理速率决定
    这里我们采用时间估计的方式，通过估计处理AI微服务所需要的时间反向推理它的处理速率
    '''
    def __init__(self, id) -> None:
        self.id = id
        self.dnn_num = random.randint(4, 6)
        self.dnn_alpha = np.random.randint(low=5, high=8, size=self.dnn_num)
        self.cpu = random.randint(5, 8)
        self.gpu = random.randint(1, 2)
        self.memory = random.randint(50, 80)

    def get_alpha(self):
        exe_time = 0
        for i in range(self.dnn_num):
            dnn_alpha = self.dnn_alpha[i]
            exe_time += 1/dnn_alpha
        return 1/exe_time
    def get_cpu(self):
        return self.cpu

    def get_gpu(self):
        return self.gpu

    def get_memory(self):
        return self.memory

class EDGE_NODE:
    '''
    边缘节点，拥有位置信息，以及资源数量
    '''
    def __init__(self, id) -> None:
        self.id = id
        self.x = random.uniform(10, 100)
        self.y = random.uniform(20, 80)
        self.cpu = random.randint(15, 25)
        self.gpu = random.randint(0,5)
        self.memory = random.randint(300,400)

    def get_location(self):
        return self.x, self.y

    def get_cpu(self):
        return self.cpu

    def get_gpu(self):
        return self.gpu

    def get_memory(self):
        return self.memory

class USER:
    '''
    用户等价与服务请求
    拥有位置和流量
    '''
    def __init__(self, id) -> None:
        self.id = id
        self.lamda = random.randint(3,10)
        self.x = random.uniform(0, 150)
        self.y = random.uniform(0, 100)

    def get_lamda(self):
        return self.lamda
    def get_location(self):
        return  self.x, self.y
    def get_request(self):
        '''
        用户会随机发出含有2-4个普通微服务和0-3个AI微服务的请求链
        :return:请求链，和用于判断微服务类型的标识符（0：普通微服务，1：AI微服务）
        '''
        request_service = []
        request_service_mark = []
        ms_list = ms_initial()
        aims_list = aims_initial()
        num_of_MS = random.randint(2, 4)
        num_of_AIMS = random.randint(0, 3)
        for _ in range(num_of_MS):
            ms = random.choice(ms_list)
            request_service.append(ms)
            request_service_mark.append(0)
            ms_list.remove(ms)
        for _ in range(num_of_AIMS):
            aims = random.choice(aims_list)
            request_service.append(aims)
            request_service_mark.append(1)
            aims_list.remove(aims)
        return request_service, request_service_mark

# ms initial
# list [MS0,MS1,...]
def ms_initial():
    ms_list = []
    for i in range(MS_NUM):
        ms_list.append(MS(i))
    return ms_list

# aims initial
# list [AIMS0,AIMS1,...]
def aims_initial():
    aims_list = []
    for i in range(AIMS_NUM):
        aims_list.append(AIMS(i))
    return aims_list

# list
# [USER0,USER1,...]
def user_initial():
    user_list = []
    for i in range(USER_NUM):
        user_list.append(USER(i))
    return user_list

# edge initial
# list[EDGE0,EDGE1,...]
def edge_initial():
    edge_node_list = []
    for i in range(NODE_NUM):
        edge_node_list.append(EDGE_NODE(i))
    return edge_node_list

# ms list
# list[a1,a2,...]
def get_ms_alpha(ms_list):
    ms_alpha_list = []
    for i in range(MS_NUM):
        ms_alpha_list.append(ms_list[i].get_alpha())
    return ms_alpha_list

# aims list
# list[a1,a2,...]
def get_aims_alpha(aims_list):
    aims_alpha_list = []
    for i in range(AIMS_NUM):
        aims_alpha_list.append(aims_list[i].get_alpha())
    return aims_alpha_list

def get_user_lamda(user):
    user_lamda_list = []
    for i in range(USER_NUM):
        user_lamda_list.append(user[i].lamda)
    return user_lamda_list

def get_user_request(user):
    '''
    user_list里面存的是对应的（用户：用户所包含的请求链）
    访问元素的时候需要用user所索引
    我们假设一个用户只会产生一个服务请求
    :param num_of_user:
    :return:
    '''
    user_list = {}
    user_request_make_list = {}
    for item in user:
        user_list[item], user_request_make_list[item] = item.get_request()
    return user, user_list, user_request_make_list

def get_bandwidth_with_node():
    '''
    每一个节点都有自己的带宽资源，随机生成
    :param node_list: 网络中节点集合
    :return:
    '''
    bandwidth = []
    for _ in range(NODE_NUM):
        b = random.randint(5, 10)
        bandwidth.append(b)
    return bandwidth

def get_data_with_ms():
    '''
    每一个节点都有自己的带宽资源，随机生成
    :param node_list: 网络中节点集合
    :return:
    '''
    data = []
    for _ in range(MS_NUM):
        d = random.randint(2, 5)
        data.append(d)
    for _ in range(AIMS_NUM):
        d = random.randint(10, 20)
        data.append(d)
    return data

def cal_dis(node1,node2):
    disx = (node1.x - node2.x) ** 2
    disy = (node1.y - node2.y) ** 2
    dis = math.sqrt(disx + disy)
    return dis

def cal_dis_user_node(user, node):
    disx = (node.x- user.x) ** 2
    disy = (node.y - user.y) ** 2
    dis = math.sqrt(disx + disy)
    return dis

def connect_nodes_within_range(nodes, initial_range=10, range_step=1):
    """
    确保节点完全连通，处理不连通的情况。
    初始距离设为10，每次扩大范围增量为1
    """
    connected_lines = []
    V = [[0] * NODE_NUM for _ in range(NODE_NUM)]
    range_factor = 1
    def is_fully_connected(lines, n):
        """
        检查图是否完全连通。
        """
        # 构建邻接表
        adjacency_list = {i: [] for i in range(n)}
        for line in lines:
            node1, node2 = line
            adjacency_list[node1].append(node2)
            adjacency_list[node2].append(node1)
        # 深度优先搜索检查连通性
        visited = set()
        stack = [0]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(adjacency_list[current])

        return len(visited) == n
    while True:
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i==j:
                    V[i][j] = 1
                if i != j:
                    dist = cal_dis(nodes[i], nodes[j])
                    if dist <= initial_range * range_factor and (i, j) not in connected_lines and (j, i) not in connected_lines:
                        if sum(V[i])>3:
                            # 减少服务器之间连接的稠密程度
                            continue
                        connected_lines.append((i, j))
                        V[i][j]=1
                        V[j][i]=1
        if is_fully_connected(connected_lines, len(nodes)):
            break  # 完全连通，结束循环
        else:
            range_factor += range_step

    return connected_lines, V

def environment_initialization():
    ms = ms_initial()  # 基础微服务
    aims = aims_initial()  # AI微服务
    all_ms = np.append(ms,aims)  # 所有的微服务集合，可以用下标来区别基础和AI
    user = user_initial()  # 用户初始化，每个用户包含一个服务请求
    node_list = edge_initial()  # 服务器初始化
    ms_alpha = get_ms_alpha(ms)  # 基础微服务处理速率
    aims_alpha = get_aims_alpha(aims)  # AI微服务处理速率
    all_ms_alpha = np.append(ms_alpha, aims_alpha)  # 所有的微服务的处理速率集合，可以用下标来区别基础和AI
    service_lamda = get_user_lamda(user)  # 用户服务请求到达率
    bandwidth = get_bandwidth_with_node()  # 服务器带宽资源
    data = get_data_with_ms()  # 微服务数据大小
    users, user_list, marker = get_user_request(user)  # 获得用户集，用户请求集，服务请求标记集
    connected_lines, graph = connect_nodes_within_range(node_list, initial_range=10)  # 初始化网络图和服务器连接
    return all_ms, all_ms_alpha, node_list, users, user_list, service_lamda, marker,\
        bandwidth, data, graph, connected_lines

if __name__ == '__main__':
    ms = ms_initial()
    aims = aims_initial()
    user = user_initial()
    node = edge_initial()
    for item in node:
        print(item.id, end=' ')
    print(' ')
    B = get_bandwidth_with_node()
    print("带宽", B)
    _,user_list, _= get_user_request(user)
    node_list = edge_initial()
    msalpha = get_ms_alpha(ms)
    aimsalpha = get_aims_alpha(aims)
    servicelamda = get_user_lamda(user)
    print("基础微服务的处理速率：", msalpha)
    print("AI微服务的处理速率：", aimsalpha)
    print("服务请求的到达率：", servicelamda)
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
        print(x,y,m)
    plt.scatter(x_list,y_list,c='red',marker='*')
    for xi, yi, mi in zip(x_list, y_list,m):
        # 给x和y添加偏移量，别和点重合在一起了
        plt.text(xi, yi + 1, mi)
    user_x_list = []
    user_y_list = []
    for i in user_list:
        x, y = i.get_location()
        user_x_list.append(x)
        user_y_list.append(y)
    # 节点连接

    connected_lines, V = connect_nodes_within_range(node_list, initial_range=10)
    print(V)
    print(connected_lines)
    for (node1, node2) in connected_lines:
        node1, node2 = node_list[node1], node_list[node2]
        plt.plot([node1.x, node2.x], [node1.y, node2.y], c='red', linestyle='-', linewidth=0.5)

    plt.scatter(user_x_list, user_y_list,c='blue')
    plt.tight_layout()
    plt.show()
