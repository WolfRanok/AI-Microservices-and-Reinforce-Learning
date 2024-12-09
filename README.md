# AI-Microservices-and-Reinforce-Learning

## 项目架构

### Baselines Algorithms

用于定义对比算法

---

### Environment
用于定义环境
#### ENV_DEF.py
服务环境：

**定义**：

1. 定义了请求链集合的生成，以及边缘节点的创建方法
2. 定义用户只发生一个请求链，用户和服务器节点都包含地理位置信息（x,y）
3. 定义每一个用户有一个请求到达率`lamda`，这个值在[3-6]之间，请求到达率只的是服务强度，即请求发送的频率

**描述**：

1. 每一个请求链有用户生成，包含若干微服务（2-4个）和AI微服务（0-3个）
2. 每一个微服务需要使用：`cpu`、`memory`(内存)
3. 每一个AI微服务需要使用：`cpu`、`gpu`、`memory`
4. 微服务和AI微服务有一定的处理速率`alpha`


**函数**：
1. 使用函数 `get_user_request` 返回：(用户对象列表, 每个用户的有且仅有一个的请求链的集合（字典）, 判定微服务是否为AI微服务的列表)
2. 使用函数 `edge_initial` 创建边缘节点的列表

#### ENV.py

**定义**:

状态: 由部署方案、资源的可用情况和剩余情况组成。

**函数**：

1. `initial_state()` : 随机初始化一个状态
2. `get_deploy(state)` : 从状态中获取部署方案
3. `get_rout(state)` : 从状态中获取路由方案
4. `get_first_node` : 获取服务请求的接收节点, 即用户请求链发送的第一个边缘节点
5. `get_ms_image(ms, aims, users, requests, marke)` : 返回微服务所需实例数，其中返回的ms_image实例镜像列表，前MS_NUM个位置存放不同普通微服务的镜像实例数，后AIMS_NUM个表示AI微服务的实例镜像分配情况


---

### Unnamed Algorithm

暂时为命名的主算法

#### network.py

定义了用于强化学习算法(Determinstic actor-critic algorithm)的两个神经网络actor和critic网络。
两个网络中都需要包含 lstm 网络，全连接层，以保证效率。
其中 actor 神经网络的定义中接受两个变量 `MA_AIMS_NUM` 和 `NODE_NUM` 两个变量，分别表示服务种类数量和节点数量。
actor 的输入层是一个规模为` 1 * (MA_AIMS_NUM * NODE_NUM + 3 * 2 * NODE_NUM)` 的一个一维向量。
输出的结果（即行动），是一个规模为 `MA_AIMS_NUM * NODE_NUM` 的一个矩阵，矩阵的每一行是一个概率选择，表示每一个微服务选择该节点的概率，所以这个矩阵的每一行的合为1

Critic 网络对每一个行动给出一个评价。

main_test.py 用于测试