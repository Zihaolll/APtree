import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from Interval_IOU import IoU, interval_set_area
from RF_to_interval import _parse_RFeachTree_interval_with_flux
from GBF_to_interval import _parse_GBFeachTree_interval_with
from Distribution_measure import distribution_measure_set
from statsmodels.distributions.empirical_distribution import ECDF


def softmax(z):
    exp_z = np.exp(z)
    softmax_output = exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)
    return softmax_output


def sigmoid(z):
    z = z.reshape(z.shape[0])
    sigmoid_output = np.array([np.exp(-z) / (1 + np.exp(-z)), 1 / (1 + np.exp(-z))])
    sigmoid_output = sigmoid_output.T
    return sigmoid_output


class ExhaustivityIntersect:
    def __init__(self, forest, left_bounds, right_bounds):
        self.forest = forest
        if isinstance(forest, RandomForestClassifier):
            self.forest_name = 'RF'
        elif isinstance(forest, GradientBoostingClassifier):
            self.forest_name = 'GBDT'

        if self.forest_name == 'RF':
            self.interval_set, self.proba_set, _ = _parse_RFeachTree_interval_with_flux(forest,
                                                                                        left_bounds=left_bounds,
                                                                                        right_bounds=right_bounds)
        elif self.forest_name == 'GBDT':
            self.interval_set, self.proba_set = _parse_GBFeachTree_interval_with(forest,
                                                                                 left_bounds=left_bounds,
                                                                                 right_bounds=right_bounds)

        self.whole_interval = np.array([left_bounds, right_bounds]).T

    def Exh_intersect_beta0(self):
        init_new_interval = self.interval_set[0].copy()
        init_new_proba = self.proba_set[0].copy()
        init_new_num = np.ones(len(init_new_proba))
        curr_new_interval = init_new_interval
        curr_new_proba = init_new_proba
        curr_new_num = init_new_num

        for k in range(1, len(self.interval_set)):
            curr_tree_interval = self.interval_set[k].copy()
            curr_tree_proba = self.proba_set[k].copy()
            new_interval = []
            new_proba = []
            new_num = []
            for i in range(len(curr_new_interval)):
                for j in range(len(curr_tree_interval)):
                    inter_interval = IoU(curr_new_interval[i], curr_tree_interval[j]).intersect()
                    if inter_interval is not None:
                        new_interval.append(inter_interval)
                        inter_num = curr_new_num[i] + 1
                        if self.forest_name == 'RF':
                            inter_proba = ((curr_new_proba[i] * curr_new_num[i]) + curr_tree_proba[j]) / inter_num
                        elif self.forest_name == 'GBDT':
                            inter_proba = curr_new_proba[i] + curr_tree_proba[j]
                        new_proba.append(inter_proba)
                        new_num.append(inter_num)
            curr_new_interval = new_interval
            curr_new_proba = new_proba
            curr_new_num = new_num

        self.curr_new_interval, self.curr_new_proba, self.curr_new_num = \
            np.array(curr_new_interval), np.array(curr_new_proba), np.array(curr_new_num)
        if self.forest_name == 'GBDT':
            if self.forest.n_classes_ == 2:
                self.curr_new_proba = sigmoid(self.curr_new_proba)
            else:
                self.curr_new_proba = softmax(self.curr_new_proba)

        return self.curr_new_interval, self.curr_new_proba, self.curr_new_num

    def Exh_intersect(self, measure, ecdf, topn=-1):
        init_new_interval = self.interval_set[0].copy()
        init_new_proba = self.proba_set[0].copy()
        init_new_num = np.ones(len(init_new_proba))
        curr_new_interval = init_new_interval
        curr_new_proba = init_new_proba
        curr_new_num = init_new_num

        for k in range(1, len(self.interval_set)):
            curr_tree_interval = self.interval_set[k].copy()
            curr_tree_proba = self.proba_set[k].copy()
            new_interval = []
            new_proba = []
            new_num = []
            for i in range(len(curr_new_interval)):
                for j in range(len(curr_tree_interval)):
                    inter_interval = IoU(curr_new_interval[i], curr_tree_interval[j]).intersect()
                    if inter_interval is not None:
                        new_interval.append(inter_interval)
                        inter_num = curr_new_num[i] + 1
                        if self.forest_name == 'RF':
                            inter_proba = ((curr_new_proba[i] * curr_new_num[i]) + curr_tree_proba[j]) / inter_num
                        elif self.forest_name == 'GBDT' or 'XGBT':
                            inter_proba = curr_new_proba[i] + curr_tree_proba[j]
                        new_proba.append(inter_proba)
                        new_num.append(inter_num)
            curr_new_interval = new_interval
            curr_new_proba = new_proba
            curr_new_num = new_num

            if measure == 'Lebesgue':

                curr_new_interval_mu = interval_set_area(
                    np.array(curr_new_interval))
            elif measure == 'Distribution':

                curr_new_interval_mu = distribution_measure_set(np.array(curr_new_interval), ecdf)

            if topn != -1 and len(curr_new_interval_mu) > topn:

                top_indices = np.argpartition(np.array(curr_new_interval_mu), -topn)[-topn:]
                t_curr_new_interval = []
                t_curr_new_proba = []
                t_curr_new_num = []
                for t_ind in top_indices:
                    t_curr_new_interval.append(curr_new_interval[t_ind])
                    t_curr_new_proba.append(curr_new_proba[t_ind])
                    t_curr_new_num.append(curr_new_num[t_ind])
                curr_new_interval = t_curr_new_interval
                curr_new_proba = t_curr_new_proba
                curr_new_num = t_curr_new_num

        self.curr_new_interval, self.curr_new_proba, self.curr_new_num = \
            np.array(curr_new_interval), np.array(curr_new_proba), np.array(curr_new_num)
        if self.forest_name == 'GBDT' or self.forest_name == 'XGBT':
            if self.forest.n_classes_ == 2:

                self.curr_new_proba = sigmoid(self.curr_new_proba)
            else:
                self.curr_new_proba = softmax(self.curr_new_proba)

        return self.curr_new_interval, self.curr_new_proba, self.curr_new_num
