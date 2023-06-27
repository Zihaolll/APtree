import numpy as np

def interval_set_to_split_point_set(interval_set):  # 选取候选区间中的所有可能切分点，同时已经去除可能的边界点
    I = interval_set
    feature_n = I.shape[1]
    sp_set = []
    for i in range(feature_n):
        point_set = np.unique(I[:, i, :])
        point_set = np.sort(point_set)[1:-2]
        for value in point_set:
            sp_set.append([i, value])
    return sp_set