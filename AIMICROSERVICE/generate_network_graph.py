import random

import numpy as np
from AIMICROSERVICE.Environment.ENV_DEF import *

x_node = np.loadtxt(open("edge_node.CSV"),delimiter=",",skiprows=1,usecols=[1])
y_node = np.loadtxt(open("edge_node.CSV"),delimiter=",",skiprows=1,usecols=[2])
print(x_node,y_node)
location_node = []
node = []
for i in range(x_node.size):
    location = (x_node[i],y_node[i])
    node.append(EDGE_NODE(i,x_node[i],y_node[i]))
    location_node.append(location)
print(location_node)
connected = []

node_10 = np.random.choice(node,10,replace=False)
for i in range(len(node_10)):
    min_dis = float("inf")
    near_node = node_10[i]
    for j in range(len(node_10)):
        if i!=j:
            dis = cal_dis(node_10[i],node_10[j])*111
            if dis<0.8:
                connected.append((node_10[i].id,node_10[j].id,dis))
    #         if dis<min_dis:
    #             min_dis=dis
    #             near_node = node_10[j]
    # connected.append((node_10[i].id,near_node.id,min_dis*111))
print(connected)
dis = cal_dis(node[7], node[i])
print(f"基站0与基站{i}之间的距离为{dis * 111}KM")
'''
    输出用户与服务器之间的位置关系图
    '''
x_list = []
y_list = []
m = []
for i in node_10:
    x_list.append(i.x)
    y_list.append(i.y)
    m.append(i.id)
    print(i.x,i.y,m)
plt.scatter(x_list,y_list,c='red',marker='*')
for xi, yi, mi in zip(x_list, y_list,m):
    # 给x和y添加偏移量，别和点重合在一起了
    plt.text(xi, yi + 1, mi)
# user_x_list = []
# user_y_list = []
# for i in user_list:
#     x, y = i.get_location()
#     user_x_list.append(x)
#     user_y_list.append(y)
# 节点连接


print(connected)
for (node1, node2,dis) in connected:
    node1, node2 = node[node1], node[node2]
    plt.plot([node1.x, node2.x], [node1.y, node2.y], c='red', linestyle='-', linewidth=0.5)

# plt.scatter(user_x_list, user_y_list,c='blue')
plt.show()


