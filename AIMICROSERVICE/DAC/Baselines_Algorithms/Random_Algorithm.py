"""
随机部署算法，每次对需求最大的微服务进行随机部署
部署时，随机选择部署的服务器节点，直到部署成功
"""
from Environment.NEW_ENV import *
from Environment.ENV_DEF import *

class Random_Algorithm:
    def option_ms(self):
        """
        选择一个微服务进行分配
        根据当前所需的实例数情况，返回最高需求量的那个
        返回-1表示已经全部部署完毕
        :return:MaxIndex
        """
        index = np.argmax(self.ms_image)
        if self.ms_image[index] == 0:
            return -1
        return index

    def refresh(self):
        """
        刷新微服务部署需求
        :return:None
        """
        self.ms_image = self.old_ms_image.copy()

    def analysis_state(self, state, index=None, flag=False):
        """
        用于分析，某状态
        :param state:
        :param flag:
        :param index: 选择部署的微服务
        :return:
        """
        print("*"*30, self.count, "*"*30)
        self.count += 1

        if flag:
            print("每一种微服务资源所需情况如下：")
            for i, s in enumerate(self.ms_aims):
                print(f"服务{i}，CPU: {s.get_cpu()}, GPU: {s.get_gpu()},内存: {s.get_memory()}")

        if index:
            s = self.ms_aims[index]
            print(f"已结束对服务 {index} 的部署: CPU: {s.get_cpu()}, GPU: {s.get_gpu()},内存: {s.get_memory()}")

        print("当前待分配实例数：\n", self.ms_image)

        print("当前的部署情况：")
        deploy = get_deploy(state).T
        print("服务类型\t \t",'\t \t'.join([f"{i}" for i in range(MA_AIMS_NUM)]))
        for i,node_deployment in enumerate(deploy):
            print(f"服务器{i}\t|\t", '\t|\t'.join([str(int(x)) for x in node_deployment]))

        print("服务器资源状态（已用|剩余|总量）：")
        deploy = get_resource(state)
        CUP = deploy[:NODE_NUM * 2]
        GUP = deploy[NODE_NUM * 2:NODE_NUM * 4]
        Memory = deploy[NODE_NUM * 4:]
        for i in range(NODE_NUM):
            print(f"服务器{i}: \tCPU：{CUP[i]} | {CUP[i+NODE_NUM]} | {CUP[i]+CUP[i+NODE_NUM]} \tGPU:{GUP[i]} | {GUP[i+NODE_NUM]} | {GUP[i]+GUP[i+NODE_NUM]} \t内存:{Memory[i]} | {Memory[i+NODE_NUM]} | {Memory[i+NODE_NUM]+Memory[i]}")
        print()

    def allocate_resources(self, index, state, node):
        """
        给某一个服务分配资源，若不够用则返回-1
        :param index: 选择的微服务类型
        :param state: 状态
        :param node: 选择部署的节点
        :return: bool 表示是否分配成功
        """
        # 分离得到部署情况和资源情况
        deploy = get_deploy(state)
        resource = get_resource(state)

        # 需要消耗的资源情况
        cpu = self.ms_aims[index].get_cpu()
        gpu = self.ms_aims[index].get_gpu()
        memory = self.ms_aims[index].get_memory()

        # 当前的资源状况
        # print(resource)
        now_cpu = resource[NODE_NUM: NODE_NUM * 2][node]
        now_gpu = resource[NODE_NUM * 3: NODE_NUM * 4][node]
        now_memory = resource[NODE_NUM * 5:][node]

        if now_cpu < cpu or now_memory < memory or now_gpu < gpu:
            # print(f'资源不够不能分配！当前CPU：{now_cpu}、GPU：{now_gpu}、内存：{now_memory}，需要的资源：CPU：{cpu}、GPU：{gpu}、内存：{memory}')
            return False

        # 可以分配资源，则开始分配
        ## 分配实例数
        # print(state)
        self.ms_image[index] -= 1
        deploy[index][node] += 1
        ## 配平相应的资源
        # cpu分配
        resource[NODE_NUM: NODE_NUM * 2][node] -= cpu
        resource[:NODE_NUM][node] += cpu
        # gpu分配
        resource[NODE_NUM * 3: NODE_NUM * 4][node] -= gpu
        resource[NODE_NUM * 2: NODE_NUM * 3][node] += gpu
        # 内存分配
        resource[NODE_NUM * 5:][node] -= memory
        resource[NODE_NUM * 4: NODE_NUM * 5][node] += memory

        return True

    def run_random_algorithm(self, state):
        """
        执行随机算法，获取部署状态
        :param state: 初试状态
        :return: state
        """
        self.state = state.copy()
        # self.analysis_state(self.state, flag=True)  # 查看一下初试状态
        self.refresh()  # 状态刷新

        count = 0  # 计数器

        while True:  # 开始部署
            # 选择待分配的节点
            index = self.option_ms()

            # 部署完成退出循环
            if index == -1:
                # print("部署完成")
                break

            # 对于指定的微服务随机选择一个节点进行部署，直到部署成功
            node_list = list(range(NODE_NUM))  # 总的服务数: MA_AIMS_NUM = MS_NUM + AIMS_NUM
            node = random.choice(node_list)

            # 针对某一个服务的部署
            while not self.allocate_resources(index, self.state, node):
                # 分配失败则重新删除节点
                node_list.remove(node)

                # 说明没有可以用的服务节点，当前节点无法完成分配
                if not node_list:
                    # print("该服务节点无法分配，其所需资源没有服务器符合要求")
                    self.ms_image[index] -= 1
                    break
                count += 1

                # 重新选择一个节点进行部署
                node = random.choice(node_list)

        # print("算法迭代次数：", count)
        # 分析部署情况
        # self.analysis_state(self.state)

        deploy = get_deploy(self.state)
        rout = get_each_request_rout(deploy)
        T = cal_total_delay(deploy, rout)
        # print("T_RA:",T)
        return T



    def __init__(self, ms_image, all_ms):
        """
        初始化需要提供，镜像需求，以及（AI）微服务类型对象
        :param ms_image:
        :param ms:
        :param aims:
        """
        self.old_ms_image = ms_image.copy()
        self.ms_image = ms_image.copy()
        self.ms_aims = all_ms
        self.count = 0 # 计数器

if __name__ == '__main__':
    # 制作一个待分配实例数
    # all_ms, all_ms_alpha, node_list, users, user_list, service_lamda, marker, bandwidth, Data, graph, connected_lines = environment_initialization()
    ms_image = get_ms_image()

    # 初始化环境
    # print(ms_image, type(ms_image))
    ra = Random_Algorithm(ms_image, all_ms)

    # 随机给出一个初始状态
    state = initial_state()

    ra.run_random_algorithm(state)