import numpy as np
from Distance_Functional import RFSimpleFuncDis
from Interval_IOU import IoU

from statsmodels.distributions.empirical_distribution import ECDF
from Distribution_measure import distribution_measure


class GainFunc:
    def __init__(self, rf, dataset_information, val_idx=None, measure_switch='Lebesgue', vector_distance_switch='NL',
                 value_space_switch='onehot', task_region=None, ecdf=None, m_topn=-1):
        self.rf = rf
        self.di = dataset_information
        self.ms = measure_switch
        self.vds = vector_distance_switch
        self.val_idx = val_idx
        self.vss = value_space_switch
        self.ecdf = ecdf
        self.m_topn = m_topn
        if self.ecdf is None:
            e_data = self.di['X'][self.val_idx]
            self.ecdf = {i: ECDF(e_data.T[i]) for i in range(e_data.shape[1])}

        self.RFSFD = RFSimpleFuncDis(rf=self.rf, dataset_information=self.di, val_idx=self.val_idx,
                                     measure_switch=self.ms, vector_distance_switch=self.vds, task_region=task_region,
                                     ecdf=self.ecdf, m_topn=self.m_topn)

    def min_dis(self, interval):
        I = interval
        D = np.zeros(self.rf.n_classes_)
        if self.vss == 'onehot':
            V_init = np.zeros(self.rf.n_classes_)
            for i in range(V_init.shape[0]):
                V = V_init.copy()
                V[i] = 1
                D[i] = self.RFSFD.calculate_dis([I], [V])
        D_min = np.min(D)
        V_best = np.zeros(self.rf.n_classes_)
        V_best[np.argmin(D)] = 1
        return D_min, V_best, D

    def gain(self, or_interval, sp_interval1, sp_interval2):  #

        if self.ms == 'Lebesgue':
            w1 = IoU(or_interval, sp_interval1).intersect_rate_for_interval1()
            w2 = IoU(or_interval, sp_interval2).intersect_rate_for_interval1()
        elif self.ms == 'Distribution':
            mu_or = distribution_measure(or_interval, self.ecdf)
            mu_sp1 = distribution_measure(sp_interval1, self.ecdf)
            mu_sp2 = distribution_measure(sp_interval2, self.ecdf)
            w1 = mu_sp1 / mu_or
            w2 = mu_sp2 / mu_or

        dm_or, P_or, _ = self.min_dis(or_interval)
        dm_sp1, P_sp1, _ = self.min_dis(sp_interval1)
        dm_sp2, P_sp2, _ = self.min_dis(sp_interval2)
        gain = dm_or - (dm_sp1 * w1) - (dm_sp2 * w2)
        return gain, P_sp1, P_sp2
