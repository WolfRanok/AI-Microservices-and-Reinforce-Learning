import numpy as np
"""
原状态构成：
state = [current_ms_info + current_request_info + deploy_state + RES + node_information + dependency + graph_1]
其中：
1. current_ms_info （ 1*5 ）：第一个微服务的 id、数据量、cpu、gpu、内存需求
2. current_request_info（1+1+1+1+MA_AIMS_NUM+MA_AIMS_NUM*NODE_NUM）: 第一个用户的 id、请求到达率、最大容忍延迟、服务请求的响应时延、待部署微服务镜像情况、路由概率表
3. deploy_state（MA_AIMS_NUM * NODE_NUM）：每一种微服务在不同服务器中的部署情况
4. RES(2 * 3 * NODE_NUM)：3种资源在不同服务器上的原有和剩余情况
5. node_information（2 * NODE_NUM）：（服务器节点id和带宽信息）*NODE_NUM
6. dependency（MA_AIMS_NUM * MA_AIMS_NUM）：生成服务器依赖图 
7. graph_1（NODE_NUM * NODE_NUM）：服务器节点的链接情况的关联矩阵


新状态构成：

"""
from ENV import *

def initstate():
    node_graph = np.reshape(graph, (NODE_NUM * NODE_NUM))
    ms_dependency = get_ms_dependency()
    deploy = np.zeros(shape=(MA_AIMS_NUM, NODE_NUM))
    resource_occ = np.zeros(shape=(NODE_NUM))