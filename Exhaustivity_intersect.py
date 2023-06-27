import numpy as np
# import cupy as np
from Interval_IOU import IoU
from RF_to_interval import _parse_RFeachTree_interval_with_flux


class ExhaustivityIntersect:
    def __init__(self, RF, left_bounds, right_bounds):
        self.interval_set, self.proba_set, self.flux_set = _parse_RFeachTree_interval_with_flux(RF,
                                                                                                left_bounds=left_bounds,
                                                                                                right_bounds=right_bounds)
        self.whole_interval = np.array([left_bounds, right_bounds]).T

    def Exh_intersect(self):
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
                        inter_proba = ((curr_new_proba[i] * curr_new_num[i]) + curr_tree_proba[j]) / inter_num
                        new_proba.append(inter_proba)
                        new_num.append(inter_num)
            curr_new_interval = new_interval
            curr_new_proba = new_proba
            curr_new_num = new_num
        self.curr_new_interval, self.curr_new_proba, self.curr_new_num = \
            np.array(curr_new_interval), np.array(curr_new_proba), np.array(curr_new_num)


        return self.curr_new_interval, self.curr_new_proba, self.curr_new_num

