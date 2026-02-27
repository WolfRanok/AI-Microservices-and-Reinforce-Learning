# AI-Microservices-and-Reinforce-Learning

## 项目简介

本项目旨在研究基于强化学习（主要为 GNN-PPO 算法）驱动的微服务智能部署与调度，结合边缘计算、AI微服务、资源分配、网络拓扑等多维度场景，支持多种算法对比、环境参数自定义、模型训练与评估、数据可视化等功能。适用于智能边缘计算、微服务调度、AI服务部署等领域的科研与工程实践。

## 主要功能

- 微服务与AI微服务的环境建模与资源分配
- 基于图神经网络的强化学习部署策略（GNN-PPO）
- 多种对比算法（DQN、FFD、随机等）与基线实现
- 环境参数可配置（服务到达率、服务长度、服务数量等）
- 支持模型训练、评估、保存与加载
- 多种数据集与模型文件管理
- 结果可视化（延迟、资源利用率等）

## 目录结构

```
AIMICROSERVICE/           # 边缘节点与环境建模相关代码
Baselines_Algorithms/     # 基线算法（FFD、随机、RLS等）
Environment/              # 环境定义与数据
MSdeploy_PPO/             # PPO部署相关代码
MSD_PPO/                  # PPO算法实现与数据
NEW_PPO/                  # GNN-PPO主实验模块（推荐入口）
Unnamed_Algorithm/        # 其他算法实验
main_test.py              # 测试入口
README.md                 # 项目说明文档
```

### NEW_PPO 主要结构

- `Agent.py`：GNN-PPO智能体定义，包含 actor/critic 网络、经验回放等
- `Arguments.py`：超参数与环境参数解析
- `Environment.py`：环境建模，用户、微服务、边缘节点、网络拓扑等
- `network.py`：图神经网络结构（GCN/GAT/LSTM等）
- `main.py`：主训练流程，模型保存、wandb日志等
- `evaluate.py`：模型评估与对比
- `plot/eva_plot.py`：结果可视化脚本
- `Data/`：训练与评估数据集
- `Model/`：模型权重文件（actor/critic）
- `Environmental_parameters/`：环境参数脚本与模型
- `users.CSV`、`edge_node.CSV`：用户与边缘节点数据

## 主要模块说明

### 环境建模（Environment.py）

- 用户、微服务、AI微服务、边缘节点、网络拓扑等对象建模
- 支持多种资源类型（CPU、GPU、内存）
- 微服务请求链生成、到达率、服务长度等参数可配置
- 网络拓扑自动生成与节点连接

### 智能体与算法（Agent.py、network.py）

- GNN-Actor/Critic 网络，支持图结构输入
- PPO 算法实现，经验回放、高回报回放、目标网络等
- 支持 DQN、FFD、随机等多种对比算法

### 参数与数据（Arguments.py、Data/、Environmental_parameters/）

- 超参数解析与环境参数脚本
- 多组训练/评估数据集，支持不同到达率、服务长度、服务数量等场景

### 训练与评估（main.py、evaluate.py）

- 支持 wandb 日志记录
- 模型保存与加载
- 评估脚本支持延迟、资源利用率等指标

### 可视化（plot/eva_plot.py）

- 支持多组实验结果的可视化
- 延迟、资源利用率等指标曲线展示

## 运行方法

1. 安装依赖（建议使用 Python 3.8+，推荐虚拟环境）
	```bash
	pip install torch torch-geometric numpy matplotlib wandb
	```
2. 配置参数（可修改 Arguments.py 或 main.py 内部参数）
3. 训练模型
	```bash
	python NEW_PPO/main.py
	```
4. 评估模型
	```bash
	python NEW_PPO/evaluate.py
	```
5. 可视化结果
	```bash
	python NEW_PPO/plot/eva_plot.py
	```

## 数据与模型说明

- `Data/`：包含多组训练与评估数据，命名区分不同参数场景
- `Model/`：保存多组 actor/critic 权重文件，命名区分不同参数场景
- `Environmental_parameters/`：支持到达率、服务长度、服务数量等多种环境参数脚本与模型

## 扩展与自定义

- 支持自定义环境参数与微服务类型
- 可扩展新的算法与网络结构
- 支持多种资源类型与节点拓扑
- 可集成更多可视化与评估指标

## 依赖列表

- Python 3.8+
- torch
- torch-geometric
- numpy
- matplotlib
- wandb

---

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