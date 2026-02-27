"""
该脚本用于计算对照算法的各种数据
对照环境与主算法统一
"""
import json
import pandas as pd
import matplotlib.pyplot as plt

from FFD import FFD_Algorithm
from Random_Algorithm import Random_Algorithm
from RLS import RLS_Algorithm

# MA_AIMS_NUM = MS_NUM + AIMS_NUM
from Environment.NEW_ENV import *
from Environment.ENV_DEF import *


def contrast_Load_balance(name="125_NODE_NUM-T and Load Balance.json"):
    """
    用于统计负载均衡的种种影响
    :return: Node
    """

    try:
        with open(rf'BA_Data/{name}', 'r+', encoding='utf-8') as f:
            lit = json.load(f)
    except FileNotFoundError:
        lit = []

    # 初始化镜像
    ms_image = get_ms_image()

    # 随机给出一个初始状态
    state = initial_state()

    # 初始化三种算法
    ffd = FFD_Algorithm(ms_image, all_ms)
    ra = Random_Algorithm(ms_image, all_ms)
    rls = RLS_Algorithm(ms_image, all_ms)

    T_ffd = ffd.run_ffd_algorithm(state)
    T_ra = ra.run_random_algorithm(state)
    T_rls = rls.run_rls_algorithm(state)

    # 计算负载均衡
    Load_ffd = cal_load_balance(ffd.state)
    Load_rls = cal_load_balance(rls.state)
    Load_ra = cal_load_balance(ra.state)

    # 计算时延

    data = {"Load_FFD": Load_ffd,
            "Load_RA": Load_ra,
            "Load_RLS": Load_rls,
            "T_ffd": T_ffd,
            "T_ra": T_ra,
            "T_rls": T_rls,
            "NODE_NUM": NODE_NUM,
            "MS_NUM": MS_NUM,
            "AIMS_NUM": AIMS_NUM,
            "USER_NUM": USER_NUM,
            }
    lit.append(data)
    print("Load_FFD:", Load_ffd, "  Load_RA:", Load_ra, "  Load_RLS:", Load_rls)

    with open(rf"BA_Data/{name}", "w", encoding='utf-8') as f:
        json.dump(lit, f, indent=4)
    print("负载均衡结果已保存")
    return lit


def xlsx2csv():
    """
    转换指定xlsx表格为csv文件
    :return:
    """
    csv = 'users_816.csv'
    xlsx = '用户位置.xlsx'
    df = pd.read_excel(xlsx, sheet_name=2)
    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(csv, index=False)
    print("保存成功！")


def show_img(url = '816_USER_NUM-T and Load Balance'):
    """
    数据可视化
    :param url:
    :return:
    """
    with open(rf'BA_Data/{url}.json', 'r', encoding='utf-8') as f:
        lit = json.load(f)
    y1, y2, y3, x = [], [], [], []
    for a in lit:
        y1.append(a["Load_FFD"])
        y2.append(a["Load_RA"])
        y3.append(a["Load_RLS"])
        x.append(a["USER_NUM"])

    # 创建一个图形
    plt.figure(figsize=(8, 6))

    # 绘制多条曲线
    plt.plot(x, y1, label='FFD', color='r', linestyle='-', linewidth=2)
    plt.plot(x, y2, label='RA', color='b', linestyle='--', linewidth=2)
    plt.plot(x, y3, label='RLS', color='g', linestyle=':', linewidth=2)

    # 添加标题和标签
    # plt.title('多条曲线示例', fontsize=16)
    plt.xlabel('user number', fontsize=14)
    plt.ylabel('load balance', fontsize=14)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 图像保存
    plt.savefig(rf'image/{url}.png', bbox_inches='tight')

    # 展示图形
    # plt.show()


if __name__ == '__main__':
    contrast_Load_balance(name="125_NODE_NUM-T and Load Balance.json")
    # xlsx2csv()
    # show_img()