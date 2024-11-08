import numpy as np


def distribution_measure(interval, ecdf):
    fp = []
    delta = 0
    for i in range(len(ecdf)):
        probs = ecdf[i]([interval[i][0], interval[i][1]])
        fp.append((probs[1] - probs[0] + delta))
    m = np.product(fp)
    return m

def distribution_measure_set(intervals, ecdf):
    fp = []
    delta = 0
    for i in range(len(ecdf)):
        probs_left = ecdf[i](intervals[:, i, 0])
        probs_right = ecdf[i](intervals[:, i, 1])
        probs = probs_right - probs_left + delta
        fp.append(probs)
    fp = np.array(fp)
    m = np.prod(fp.T, axis=1)
    return m