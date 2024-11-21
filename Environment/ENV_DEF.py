import random
import math
import matplotlib.pyplot as plt
import numpy as np

MS_NUM = 5
AIMS_NUM = 3
NODE_NUM = 5
USER_NUM =3
RESOURCE = 3

random.seed(123)
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
        self.cpu = random.randint(15, 25) * 5
        self.gpu = random.randint(0,5) * 5
        self.memory = random.randint(300,400) * 5

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
        self.lamda = random.randint(3, 6)
        self.x = random.uniform(0, 150)
        self.y = random.uniform(0, 100)

    def get_lamda(self):
        return self.lamda
    def get_location(self):
        return  self.x, self.y
    def get_request(self):
        """
        用户会随机发出含有2-4个普通微服务和0-3个AI微服务的请求链
        :return:请求链，和用于判断微服务类型的标识符（0：普通微服务，1：AI微服务）
        """
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

if __name__ == '__main__':
    ms = ms_initial()
    aims = aims_initial()
    user = user_initial()
    node = edge_initial()
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
    for i in node_list:
        x, y = i.get_location()
        x_list.append(x)
        y_list.append(y)
    plt.scatter(x_list,y_list,c='red',marker='*')
    user_x_list = []
    user_y_list = []
    for i in user_list:
        x, y = i.get_location()
        user_x_list.append(x)
        user_y_list.append(y)
    plt.scatter(user_x_list, user_y_list,c='blue')
    plt.tight_layout()
    plt.show()
