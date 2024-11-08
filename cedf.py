from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np

def set_ecdf(data):
    ecdf_dict = {i: ECDF(data.T[i]) for i in range(data.shape[1])}
    return ecdf_dict


def calculate_branch_probability_by_ecdf(interval, ecdf):
    features_probabilities = []
    delta = 0.000000001
    for i in range(len(ecdf)):
        probs = ecdf[i]([interval[i][0], interval[i][1]])
        print(probs)
        features_probabilities.append((probs[1] - probs[0] + delta))
    return np.product(features_probabilities)

