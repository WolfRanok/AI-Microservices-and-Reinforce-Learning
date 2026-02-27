import random
import numpy as np

class MS:
    '''
    基础微服务拥有两种资源类型
    '''
    def __init__(self, id) -> None:
        self.id = id
        # self.alpha = random.randint(10, 20)
        # self.alpha = random.randint(5, 10)
        # self.alpha = random.randint(4, 8) ## 修改之前
        self.alpha = random.randint(7, 10)

        # self.cpu = random.randint(1, 2)
        self.cpu = 1
        # self.memory = random.randint(10, 15)
        self.memory = 13
        # self.data = random.randint(2, 5)
        self.data = 1
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
        self.dnn_num = random.randint(2, 3)
        self.dnn_alpha = np.random.randint(low=20, high=30, size=self.dnn_num)
        # self.dnn_alpha = np.random.randint(low=20, high=30, size=self.dnn_num)
        # self.cpu = random.randint(5, 8)
        # self.gpu = random.randint(1, 2)
        self.cpu = 3
        self.gpu = 1
        # self.memory = random.randint(50, 80)
        self.memory = 65
        # self.data = random.randint(5, 10)
        self.data = 3
        self.alpha = self.get_alpha()
    def get_alpha(self):
        exe_time = 0
        for i in range(self.dnn_num):
            dnn_alpha = self.dnn_alpha[i]
            exe_time += 1/dnn_alpha
        return round(1/exe_time,4)
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
    def __init__(self, id,x,y,gpu) -> None:
        self.id = id
        # self.x = random.uniform(10, 100)
        # self.y = random.uniform(20, 80)
        self.x = x
        self.y = y
        self.gpu = gpu
        # self.bandwidth = random.randint(5, 20)
        # if gpu==0:
        #     self.cpu = random.randint(12, 24)
        #     self.memory = random.randint(400, 500)
        # else:
        #     self.cpu = random.randint(48, 64)
        #     self.memory = random.randint(800, 900)
        if gpu==0:
            # self.cpu = random.randint(12, 24)
            # self.memory = random.randint(200, 250)
            self.cpu = 20
            self.memory = 250
            self.bandwidth = random.randint(20, 30)
        else:
            # self.cpu = random.randint(24, 36)
            # self.memory = random.randint(400, 450)
            self.cpu = 30
            self.memory = 500
            self.bandwidth = random.randint(40, 60)


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
    def __init__(self, id, x, y, ms_list, aims_list, min_arrival, max_arrival, AI_service_num,
                 min_length_of_f_service, max_length_of_f_service,
                 min_length_of_f_service_in_ai, max_length_of_f_service_in_ai,
                 min_length_of_ai_service_in_ai, max_length_of_ai_service_in_ai) -> None:
        self.id = id
        # self.lamda = random.randint(10,15) ## 修改之前
        self.lamda = random.uniform(min_arrival, max_arrival)
        self.request_data = random.randint(10,15)
        self.x = x
        self.y = y
        if self.id >= AI_service_num:
            self.request_chain = self.get_func_request(ms_list, min_length_of_f_service, max_length_of_f_service)
        else:
            self.request_chain = self.get_AI_request(ms_list,aims_list,min_length_of_f_service_in_ai, max_length_of_f_service_in_ai,
                                                     min_length_of_ai_service_in_ai, max_length_of_ai_service_in_ai)
    def get_lamda(self):
        return self.lamda
    def get_location(self):
        return  self.x, self.y
    def get_func_request(self,ms_list, min_length_of_f_service, max_length_of_f_service):
        '''
        用户会随机发出含有2-4个普通微服务和0-3个AI微服务的请求链
        :return:请求链，和用于判断微服务类型的标识符（0：普通微服务，1：AI微服务）
        '''
        ms_list_copies = ms_list.copy()
        request_service = []
        num_of_MS = random.randint(min_length_of_f_service, max_length_of_f_service)
        for _ in range(num_of_MS):
            ms = random.choice(ms_list_copies)
            request_service.append(ms)
            ms_list_copies.remove(ms)
        return request_service
    def get_AI_request(self,ms_list,aims_list,min_length_of_f_service_in_ai, max_length_of_f_service_in_ai,
                       min_length_of_ai_service_in_ai, max_length_of_ai_service_in_ai):
        '''
        用户会随机发出含有2-4个普通微服务和0-3个AI微服务的请求链
        :return:请求链，和用于判断微服务类型的标识符（0：普通微服务，1：AI微服务）
        '''
        ms_list_copies = ms_list.copy()
        aims_list_copies = aims_list.copy()
        request_service = []
        num_of_MS = random.randint(min_length_of_f_service_in_ai, max_length_of_f_service_in_ai)
        num_of_AIMS = random.randint(min_length_of_ai_service_in_ai, max_length_of_ai_service_in_ai)
        for _ in range(num_of_AIMS):
            aims = random.choice(aims_list_copies)
            request_service.append(aims)
            aims_list_copies.remove(aims)
        for _ in range(num_of_MS):
            ms = random.choice(ms_list_copies)
            request_service.append(ms)
            ms_list_copies.remove(ms)
        return request_service