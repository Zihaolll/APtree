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



if __name__ == '__main__':
    import joblib
    import os
    from RF_to_interval import _parse_RFeachTree_interval_with_flux


    dataset_list = ['iris', 'aust_credit', 'banknote', 'haberman', 'mammographic',
                    'breast_cancer', 'heloc', 'wine_data', 'liver', 'bank', '2D_data_sin']
    path = 'E1_Exp_Task_3_5'
    for dataset_name in ['compas']:
        load_path_information = os.path.join(path, dataset_name, 'dataset_information.pkl')
        dataset_information = joblib.load(load_path_information)
        for i in range(1):
            load_path_bbx = os.path.join(path, dataset_name, '%d' % i)
            train_idx = np.load(os.path.join(load_path_bbx, 'train.npy'), allow_pickle=True)
            val_idx = np.load(os.path.join(load_path_bbx, 'val.npy'), allow_pickle=True)
            test_idx = np.load(os.path.join(load_path_bbx, 'test.npy'), allow_pickle=True)
            rf = joblib.load(os.path.join(load_path_bbx, 'rf.pkl'))
        train_X = dataset_information['X'][train_idx]
        ecdf = set_ecdf(train_X)
        II, PP, _ = _parse_RFeachTree_interval_with_flux(rf, left_bounds=dataset_information['space_lower_bound'],
                                                          right_bounds=dataset_information[
                                                              'space_upper_bound'])
        P = []
        # task_region = np.array([dataset_information['space_lower_bound'], dataset_information['space_upper_bound']]).T
        # pp = calculate_branch_probability_by_ecdf(task_region, ecdf)
        for i in range (II[0].shape[0]):
            P.append(calculate_branch_probability_by_ecdf(II[0][i], ecdf))
