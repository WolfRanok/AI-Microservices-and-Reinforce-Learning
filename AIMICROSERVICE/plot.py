"""
训练结果可视化脚本
"""
import numpy as np
import matplotlib.pyplot as plt

reward = np.loadtxt(open("DAC/Data/2025_02_04_data_10_15_3_new_state.csv"),delimiter=",",skiprows=1,usecols=[1])
param_change = np.loadtxt(open("DAC/Data/2025_02_04_data_10_15_3_new_state.csv"),delimiter=",",skiprows=1,usecols=[3])
T = np.loadtxt(open("DAC/Data/2025_02_04_data_10_15_3_new_state.csv"),delimiter=",",skiprows=1,usecols=[4])

x = np.arange(1,101)
plt.plot(x[1:],reward[1:])
plt.show()
plt.plot(x[1:],param_change[1:])
plt.show()
plt.plot(x[1:],T[1:])
plt.show()