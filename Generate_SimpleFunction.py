import numpy as np


class Generate_SimpleFunction:
    def __init__(self, init_interval, split_point_set, max_step, gainfunction):
        self.init_I = init_interval
        self.sp_set = split_point_set
        self.max_step = max_step
        self.gainfunc = gainfunction
        _, self.I_value, self.I_dis = self.gainfunc.min_dis(self.init_I)
        self.I_dis = (sum(self.I_dis) - self.I_dis) / sum(sum(self.I_dis) - self.I_dis)

    def split(self, curr_step):
        self.curr_step = curr_step
        if len(self.sp_set) == 0:
            self.left = None
            self.right = None
            return
        self.split_information = self.find_sp(self.sp_set, self.init_I, sample_num=len(self.sp_set))

        self.split_feature = self.split_information['feature id']
        self.split_value = self.split_information['value']
        self.split_gain = self.split_information['gain']

        is_splitable = self.is_splitbale()
        if is_splitable == False:
            self.left = None
            self.right = None
            return

        left_sp_set, right_sp_set = self.sp_set_update(self.sp_set, self.split_feature, self.split_value)

        self.left = Generate_SimpleFunction(init_interval=self.split_information['left interval'],
                                            split_point_set=left_sp_set,
                                            max_step=self.max_step,
                                            gainfunction=self.gainfunc)
        self.left.split(self.curr_step + 1)

        self.right = Generate_SimpleFunction(init_interval=self.split_information['right interval'],
                                             split_point_set=right_sp_set,
                                             max_step=self.max_step,
                                             gainfunction=self.gainfunc)
        self.right.split(self.curr_step + 1)

    def find_sp(self, sp_set, init_I, sample_num=10):
        np.random.seed(0)
        num_max = len(sp_set)
        sample_idx = np.arange(num_max)
        np.random.shuffle(sample_idx)
        gain = 0
        i = 0
        split_information = []
        num_bound = np.min((sample_num, num_max))

        while (gain == 0 or i < num_bound) and i < num_max:

            sp = sp_set[sample_idx[i]]
            feature_id = int(sp[0])
            value = sp[1]
            curr_split_information = self.get_split_information(feature_id, value, init_I)
            curr_gain = curr_split_information['gain']

            if gain <= curr_gain:
                gain = curr_gain
                split_information = curr_split_information
            i += 1

        return split_information

    def get_split_information(self, feature_id, value, init_I):
        split_information = {}

        left_I = init_I.copy()
        right_I = init_I.copy()

        left_I[feature_id][1] = value
        right_I[feature_id][0] = value

        gain, left_v, right_v = self.gainfunc.gain(init_I, left_I, right_I)

        split_information['gain'] = gain
        split_information['feature id'] = feature_id
        split_information['value'] = value
        split_information['left interval'] = left_I
        split_information['left vector'] = left_v
        split_information['right interval'] = right_I
        split_information['right vector'] = right_v

        return split_information

    def sp_set_update(self, sp_set, split_feature, split_value):
        sp_set = np.array(sp_set)
        left_sp_set = sp_set.copy()
        right_sp_set = sp_set.copy()

        feature_mask = sp_set[:, 0] != split_feature
        left_value_mask = sp_set[:, 1] <= split_value
        right_value_mask = sp_set[:, 1] > split_value
        left_mask = np.logical_or(feature_mask, left_value_mask)
        right_mask = np.logical_or(feature_mask, right_value_mask)

        left_sp_set = (left_sp_set[left_mask]).tolist()
        right_sp_set = (right_sp_set[right_mask]).tolist()

        return left_sp_set, right_sp_set

    def is_splitbale(self):
        if self.curr_step >= self.max_step or self.split_gain == 0:
            return False

    def value_and_step_count(self, x):
        if self.left is None and self.right is None:
            return self.I_value, self.I_dis, 1
        if x[self.split_feature] <= self.split_value:
            value, dis, step_count = self.left.value_and_step_count(x)
            return value, dis, step_count + 1
        else:
            value, dis, step_count = self.right.value_and_step_count(x)
            return value, dis, step_count + 1

    def get_interval_and_value(self):
        interval_set = []
        value_set = []
        dis_set = []
        if self.left == None and self.right == None:
            interval_set.append(self.init_I)
            value_set.append(self.I_value)
            dis_set.append(self.I_dis)
            return interval_set, value_set, dis_set
        else:
            left_interval_set, left_value_set, left_dis_set = self.left.get_interval_and_value()
            right_interval_set, right_value_set, right_dis_set = self.right.get_interval_and_value()
            interval_set = left_interval_set + right_interval_set
            value_set = left_value_set + right_value_set
            dis_set = left_dis_set + right_dis_set
            return interval_set, value_set, dis_set

    def pruning(self):
        self.is_last_node = False
        if self.left is None:
            self.is_last_node = True
            return self.is_last_node
        self.left.is_last_node = self.left.pruning()
        self.right.is_last_node = self.right.pruning()
        if self.left.is_last_node and self.right.is_last_node:
            if np.argmax(self.left.I_value) == np.argmax(self.right.I_value):
                self.left = None
                self.right = None
                self.is_last_node = True
            return self.is_last_node

    def cutting(self, step_bound):

        if self.curr_step < step_bound and self.left is not None and self.right is not None:
            self.left.cutting(step_bound)
            self.right.cutting(step_bound)
        else:
            self.left = None
            self.right = None
            return
