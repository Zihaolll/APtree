from Explainer import Explain_RF_via_SimpleFunction
import numpy as np
from sklearn import metrics
import time


class Main:
    def __init__(self, rf, dataset_information, max_depth, test_idx, alpha=0, SP_size=10, val_idx=None,
                 task_region=None,
                 measure='Distribution',
                 vector_distance='NL',
                 ecdf=None):
        self.rf = rf
        self.di = dataset_information
        self.max_depth = max_depth
        self.alpha = alpha
        self.SP_size = SP_size
        self.test_idx = test_idx
        self.val_idx = val_idx
        self.task_region = task_region
        self.ms = measure
        self.vds = vector_distance
        self.ecdf = ecdf

        self.test_X = self.di['X'][self.test_idx]
        self.test_Y = self.di['Y'][self.test_idx]
        self.rf_Y = self.rf.predict(self.test_X)
        self.rf_accuracy = metrics.accuracy_score(self.test_Y, self.rf_Y)
        self.rf_f1_score = metrics.f1_score(self.test_Y, self.rf_Y, average='macro')

    def run(self):
        ANS = {}
        P_V = {}
        P_C = {}
        param = self.ms + '-' + self.vds
        Explainer = Explain_RF_via_SimpleFunction(rf=self.rf,
                                                  dataset_information=self.di,
                                                  val_idx=self.val_idx,
                                                  measure_switch=self.ms,
                                                  alpha=self.alpha,
                                                  SP_size=self.SP_size,
                                                  vector_distance_switch=self.vds,
                                                  ecdf=self.ecdf)
        begin_time = time.time()
        APtree = Explainer.generate_explanation(self.max_depth)
        APtree.pruning()
        end_time = time.time()

        result, p_v, p_c = self.analysis_SF(APtree, self.max_depth)
        P_V[param + '_%d' % self.max_depth] = p_v
        P_C[param + '_%d' % self.max_depth] = p_c
        result['APtree_running_time'] = end_time - begin_time
        result['rf_size'] = Explainer.gainfunc.RFSFD.intervals.shape[0]
        result['rf_accuracy'] = self.rf_accuracy
        result['rf_f1_score'] = self.rf_f1_score
        ANS[param] = result
        return ANS, P_V, P_C

    def analysis_SF(self, SimpleFunction, cut_param):
        SF_result = {}
        SF = SimpleFunction
        SF_Iset, SF_Vset = SF.get_interval_and_value()
        SF_Y_class = []
        SF_Y_vector = []
        SF_Y_depth = []  # 每次预测需要划分几个节点
        for i in range(self.test_X.shape[0]):
            x = self.test_X[i]
            v, depth = SF.value_and_depth_count(x)
            SF_Y_vector.append(v)
            SF_Y_class.append(self.rf.classes_[np.argmax(v)])
            SF_Y_depth.append(depth)
        if cut_param is None:
            cp = '_'
        else:
            cp = '_%d' % cut_param

        SF_result['APtree_size' + cp] = len(SF_Iset)
        SF_result['APtree_fidelity_accuracy' + cp] = metrics.accuracy_score(self.rf_Y, SF_Y_class)
        SF_result['APtree_fidelity_f1' + cp] = metrics.f1_score(self.rf_Y, SF_Y_class, average='macro')

        SF_result['APtree_predict_accuracy' + cp] = metrics.accuracy_score(self.test_Y, SF_Y_class)
        SF_result['APtree_predict_f1' + cp] = metrics.f1_score(self.test_Y, SF_Y_class, average='macro')

        return SF_result, SF_Y_vector, SF_Y_class
