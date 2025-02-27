"""
该脚本用于计算对照算法的各种数据
对照环境与主算法统一
"""
import json

from FFD import FFD_Algorithm
from Random_Algorithm import Random_Algorithm
from RLS import RLS_Algorithm

# MA_AIMS_NUM = MS_NUM + AIMS_NUM
from Environment.NEW_ENV import *
from Environment.ENV_DEF import *


def contrast_Load_balance():
    """
    用于统计负载均衡的种种影响
    :return: Node
    """
    name = "Request_Length-Load Balance.json"

    with open(rf'BA_Data/{name}', 'r', encoding='utf-8') as f:
        lit = json.load(f)

    # 初始化镜像
    ms_image = get_ms_image()

    # 随机给出一个初始状态
    state = initial_state()

    # 初始化三种算法
    ffd = FFD_Algorithm(ms_image, all_ms)
    ra = Random_Algorithm(ms_image, all_ms)
    rls = RLS_Algorithm(ms_image, all_ms)

    ffd.run_ffd_algorithm(state)
    ra.run_random_algorithm(state)
    rls.run_rls_algorithm(state)

    Load_ffd = cal_load_balance(ffd.state)
    Load_rls = cal_load_balance(rls.state)
    Load_ra = cal_load_balance(ra.state)

    data = {"Load_FFD": Load_ffd,
            "Load_RA": Load_ra,
            "Load_RLS": Load_rls,
            "NODE_NUM": NODE_NUM,
            "MS_NUM": MS_NUM,
            "AIMS_NUM": AIMS_NUM,
            "USER_NUM": USER_NUM,
            "RESOURCE": USER_NUM,
            }
    lit.append(data)
    print("Load_FFD:", Load_ffd, "  Load_RA:", Load_ra, "  Load_RLS:", Load_rls)

    with open(rf"BA_Data/{name}", "w", encoding='utf-8') as f:
        json.dump(lit, f, indent=4)
    print("负载均衡结果已保存")
    return lit

MA_AIMS_NUM = MS_NUM + AIMS_NUM



if __name__ == '__main__':
    contrast_Load_balance()
