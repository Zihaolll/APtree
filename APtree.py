from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from Explainer import Explain_RF_via_SimpleFunction
import numpy as np
from sklearn import metrics
import time
from GBF_to_interval import _parse_GBF_interval


class APtree:
    def __init__(self, TreeEnsemble, dataset_information, max_step, test_idx, val_idx=None, task_region=None,
                 measure='Distribution', vector_distance='NL', ecdf=None):
        self.rf = TreeEnsemble
        self.TE = TreeEnsemble
        self.di = dataset_information
        self.max_step = max_step
        self.test_idx = test_idx
        self.val_idx = val_idx
        self.task_region = task_region
        self.ms = measure
        self.vds = vector_distance
        self.ecdf = ecdf

        self.test_X = self.di['X'][self.test_idx]
        self.test_Y = self.di['Y'][self.test_idx]
        self.TE_Y = self.TE.predict(self.test_X)
        self.TE_accuracy = metrics.accuracy_score(self.test_Y, self.TE_Y)
        self.TE_f1_score = metrics.f1_score(self.test_Y, self.TE_Y, average='macro')

        if isinstance(self.TE, RandomForestClassifier):
            self.forest_name = 'RF'
        elif isinstance(self.TE, GradientBoostingClassifier):
            self.forest_name = 'GBDT'
            self.vds = 'CL'


    def run(self):
        Explainer = Explain_RF_via_SimpleFunction(rf=self.TE,
                                                  dataset_information=self.di,
                                                  val_idx=self.val_idx,
                                                  measure_switch=self.ms,
                                                  vector_distance_switch=self.vds,
                                                  ecdf=self.ecdf)
        begin_time = time.time()
        SF = Explainer.generate_explanation(self.max_step)
        SF.pruning()
        end_time = time.time()

        result, p_v, p_c = self.analysis_SF(SF)
        result['running_time'] = end_time - begin_time
        if self.forest_name == 'RF':
            result['TE_partitions_n'] = Explainer.gainfunc.RFSFD.intervals.shape[0]
        elif self.forest_name == 'GBDT':
            Is, Ps = _parse_GBF_interval(self.TE, self.di['space_lower_bound'],
                                         self.di['space_upper_bound'])
            result['TE_partitions_n'] = Is.shape[0]
        result['TE_accuracy'] = self.TE_accuracy
        result['TE_f1_score'] = self.TE_f1_score
        return result, p_v, p_c

    def analysis_SF(self, SimpleFunction):
        SF_result = {}
        SF = SimpleFunction
        SF_Iset, SF_Vset, SF_Dset = SF.get_interval_and_value()
        SF_Y_class = []
        SF_Y_vector = []
        SF_Y_step = []  # 每次预测需要划分几个节点
        for i in range(self.test_X.shape[0]):
            x = self.test_X[i]
            v, _, step = SF.value_and_step_count(x)
            SF_Y_vector.append(v)
            SF_Y_class.append(self.TE.classes_[np.argmax(v)])
            SF_Y_step.append(step)

        SF_result['partitions_n'] = len(SF_Iset)

        SF_result['fidelity_accuracy'] = metrics.accuracy_score(self.TE_Y, SF_Y_class)
        SF_result['fidelity_f1'] = metrics.f1_score(self.TE_Y, SF_Y_class, average='macro')

        SF_result['predict_accuracy'] = metrics.accuracy_score(self.test_Y, SF_Y_class)
        SF_result['predict_f1'] = metrics.f1_score(self.test_Y, SF_Y_class, average='macro')

        return SF_result, SF_Y_vector, SF_Y_class

