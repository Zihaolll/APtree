from Generate_SimpleFunction import Generate_SimpleFunction
from Gain_Function import GainFunc
from Get_SplitPoint_Set import interval_set_to_split_point_set
import numpy as np
from RF_to_interval import _parse_forest_interval


class Explain_RF_via_SimpleFunction:
    def __init__(self, rf, dataset_information, val_idx=None, measure_switch='Lebesgue', vector_distance_switch='NL',
                 value_space_switch='onehot', alpha=0, SP_size=10, task_region=None, ecdf=None):
        self.rf = rf
        self.di = dataset_information
        self.val_idx = val_idx
        self.ms = measure_switch
        self.vds = vector_distance_switch
        self.vss = value_space_switch
        self.ecdf = ecdf
        self.alpha = alpha
        self.SP_size = SP_size
        if task_region is not None:
            self.task_region == task_region
        else:
            self.task_region = np.array([self.di['space_lower_bound'], self.di['space_upper_bound']]).T

        self.gainfunc = GainFunc(rf=self.rf, dataset_information=self.di, val_idx=self.val_idx,
                                 measure_switch=self.ms, vector_distance_switch=self.vds,
                                 value_space_switch=self.vss, ecdf=self.ecdf)
        interval_set, _ = _parse_forest_interval(rf, left_bounds=self.di['space_lower_bound'],
                                                 right_bounds=self.di['space_upper_bound'])
        self.sp_set = interval_set_to_split_point_set(interval_set)

    def generate_explanation(self, max_depth):
        SF = Generate_SimpleFunction(init_interval=self.task_region, split_point_set=self.sp_set,
                                     max_depth=max_depth, gainfunction=self.gainfunc,
                                     alpha=self.alpha, SP_size=self.SP_size)
        SF.split(0)
        return SF
