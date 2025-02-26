"""
该脚本用于计算对照算法的各种数据
对照环境与主算法统一
"""
import json

from FFD import FFD_Algorithm
from Random_Algorithm import Random_Algorithm
from RLS import RLS_Algorithm

from AIMICROSERVICE.Environment.NEW_ENV import *
from AIMICROSERVICE.Environment.ENV_DEF import *
MA_AIMS_NUM = MS_NUM + AIMS_NUM


def contrast_T():
    """
    用于计算几种算法的时延
    :return: Node
    """
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

    data = {"T_ffd": T_ffd, "T_ra": T_ra, "T_rls": T_rls}

    with open(rf"BA_Data/{NODE_NUM}_{MS_NUM}_{AIMS_NUM}_T.json","w",encoding='utf-8') as f:
        json.dump(data, f,indent=4)
    print("时延结果已保存")
    return data

if __name__ == '__main__':
    contrast_T()