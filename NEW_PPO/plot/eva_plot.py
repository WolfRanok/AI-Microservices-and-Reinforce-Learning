import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
df = np.genfromtxt(
    '/project/CY/Data/SIL_model_evaluation_data_1e-5_1e-5',
    delimiter=',',       # 分隔符
    names=True,# 将第一行作为列名
    dtype=None,          # 自动推断数据类型
    encoding='utf-8'     # 处理文本编码（默认可能报错）
)
df2 = np.genfromtxt(
    '/project/CY/Data/SIL_model_evaluation_data_5e-5_1e-5',
    delimiter=',',       # 分隔符
    names=True,# 将第一行作为列名
    dtype=None,          # 自动推断数据类型
    encoding='utf-8'     # 处理文本编码（默认可能报错）
)
df3 = np.genfromtxt(
    '/project/JZ/Data/SIL_model_evaluation_data_5e-4_1e-4',
    delimiter=',',       # 分隔符
    names=True,# 将第一行作为列名
    dtype=None,          # 自动推断数据类型
    encoding='utf-8'     # 处理文本编码（默认可能报错）
)
df4 = np.genfromtxt(
    '/project/YZL/Data/SIL_model_evaluation_data_1e-4_1e-4',
    delimiter=',',       # 分隔符
    names=True,# 将第一行作为列名
    dtype=None,          # 自动推断数据类型
    encoding='utf-8'     # 处理文本编码（默认可能报错）
)

df5 = np.genfromtxt(
    '/project/CY/Data/model_evaluation_data_5e-5_1e-5',
    delimiter=',',       # 分隔符
    names=True,# 将第一行作为列名
    dtype=None,          # 自动推断数据类型
    encoding='utf-8'     # 处理文本编码（默认可能报错）
)
without_sil = np.genfromtxt(
    '/project/CY/Data/model_training_data_5e-5_1e-5',
    delimiter=',',       # 分隔符
    names=True,# 将第一行作为列名
    dtype=None,          # 自动推断数据类型
    encoding='utf-8'     # 处理文本编码（默认可能报错）
)

# 选择名为 'temperature' 的列
d = df['evaluate_reward']
d2 = df2['evaluate_reward']
d3 = df3['evaluate_reward']
d4 = df4['evaluate_reward']
d5 = df5['evaluate_reward']

# 绘制温度数据的直方图
plt.figure()
plt.plot(d, color ='royalblue',linestyle='-',label = "lr=1e-5")
plt.plot(d2, color ='g',linestyle='-',label = "lr=5e-5")
plt.plot(d4, color ='red',linestyle='-',label = "lr=1e-4")
plt.plot(d3, color ='y',linestyle='-',label = "lr=5e-4")
# plt.plot(d5, color ='y',linestyle='-',label = "without_SIL")

plt.title('evaluate_rewards')
plt.legend(loc = 'lower right',prop = {'size':12})
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()