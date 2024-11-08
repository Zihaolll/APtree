import numpy as np
import sys
def interval_area(interval):
    I = interval.copy()
    interval_length = I[:, 1] - I[:, 0] + 1e-5
    area = np.prod(interval_length)
    return area


def interval_set_area(interval_set):
    I_set = interval_set.copy()
    interval_set_length = I_set[:, :, 1] - I_set[:, :, 0] + 1e-5
    area_set = np.prod(interval_set_length, axis=1)
    return area_set

class IoU():
    def __init__(self, Interval_1, Interval_2):
        self.interval_1 = Interval_1.copy()
        self.interval_2 = Interval_2.copy()
        self.interval_1_area = interval_area(self.interval_1)
        self.interval_2_area = interval_area(self.interval_2)
        self.left_max = np.max((self.interval_1[:, 0], self.interval_2[:, 0]), axis=0)
        self.left_min = np.min((self.interval_1[:, 0], self.interval_2[:, 0]), axis=0)
        self.right_max = np.max((self.interval_1[:, 1], self.interval_2[:, 1]), axis=0)
        self.right_min = np.min((self.interval_1[:, 1], self.interval_2[:, 1]), axis=0)
        self.intersect_mask = (self.left_max < self.right_min) + 0
        self.intersect_none_mask = (self.left_max >= self.right_min) + 0

    def intersect(self):
        if np.prod(self.intersect_mask) != 0:
            self.intersect_interval = np.zeros(shape=self.interval_1.shape)
            self.intersect_interval[:, 0] = self.left_max
            self.intersect_interval[:, 1] = self.right_min
        else:
            self.intersect_interval = None
        return self.intersect_interval

    def intersect_area(self):
        self.intersect()
        if self.intersect_interval is not None:
            area = interval_area(self.intersect_interval)
        else:
            area = 0
        return area

    def iou(self):
        self.intersect()
        if self.intersect_interval is not None:
            self.iou = interval_area(self.intersect_interval) / (self.interval_1_area + self.interval_2_area - interval_area(self.intersect_interval))
        else:
            self.iou = 0
        return self.iou

    def intersect_rate_for_interval1(self):
        self.intersect()
        if self.intersect_interval is not None:
            self.area_intersect = interval_area(self.intersect_interval)
            ans = self.area_intersect/self.interval_1_area
        else:
            ans = 0
        return ans

    def intersect_rate_for_interval2(self):
        self.intersect()
        if self.intersect_interval is not None:
            self.area_intersect = interval_area(self.intersect_interval)
            ans = self.area_intersect / self.interval_2_area
        else:
            ans = 0
        return ans

    def ioa_max(self):
        ans_1 = self.intersect_rate_for_interval1()
        ans_2 = self.intersect_rate_for_interval2()
        ans = max(ans_1, ans_2)
        return ans

    def rule_union(self):
        if np.prod(self.intersect_mask) != 0:
            self.rule_union_interval = np.zeros(shape=self.interval_1.shape)
            self.rule_union_interval[:, 0] = self.left_min
            self.rule_union_interval[:, 1] = self.right_max
        else:
            self.rule_union_interval = None
        return self.rule_union_interval

    def rule_cut(self, switch):
        self.intersect()
        if self.intersect_interval is not None:
            if switch == '1':
                minuend_interval = self.interval_1.copy()
            elif switch == '2':
                minuend_interval = self.interval_2.copy()

            length_minuend = minuend_interval[:, 1] - minuend_interval[:, 0]
            length_left_remainder = self.intersect_interval[:, 0] - minuend_interval[:, 0]
            length_right_remainder = minuend_interval[:, 1] - self.intersect_interval[:, 1]
            length_left_remainder_rate = length_left_remainder / length_minuend
            length_right_remainder_rate = length_right_remainder / length_minuend
            remainder_interval = minuend_interval.copy()
            if np.max(length_left_remainder_rate) > np.max(length_right_remainder_rate):
                cut_feature_id = np.argmax(length_left_remainder_rate)
                remainder_interval[cut_feature_id, 1] = self.intersect_interval[cut_feature_id, 0]
            else:
                cut_feature_id = np.argmax(length_right_remainder_rate)
                remainder_interval[cut_feature_id, 0] = self.intersect_interval[cut_feature_id, 1]
        else:
            sys.exit()
        return remainder_interval

class IoU_set():
    def __init__(self, Interval_1, Interval_set):
        self.interval_1 = Interval_1.copy()
        self.interval_set = Interval_set.copy()
        self.interval_1_set = np.tile(self.interval_1, (self.interval_set.shape[0], 1)).reshape(self.interval_set.shape)
        self.interval_1_area = interval_area(self.interval_1)
        self.left_max = np.max((self.interval_1_set[:, :, 0], self.interval_set[:, :, 0]), axis=0)
        self.left_min = np.min((self.interval_1_set[:, :, 0], self.interval_set[:, :, 0]), axis=0)
        self.right_max = np.max((self.interval_1_set[:, :, 1], self.interval_set[:, :, 1]), axis=0)
        self.right_min = np.min((self.interval_1_set[:, :, 1], self.interval_set[:, :, 1]), axis=0)
        self.intersect_mask = (self.left_max < self.right_min) + 0
        self.intersect_none_mask = (self.left_max >= self.right_min) + 0


    def intersect(self):
        self.none_mask = np.prod(self.intersect_mask, axis=1) == 0
        self.intersect_set = np.zeros(shape=self.interval_set.shape)
        self.intersect_set[:, :, 0] = self.left_max
        self.intersect_set[:, :, 1] = self.right_min
        self.intersect_set[self.none_mask] = None
        return self.intersect_set

    def intersect_rate_for_interval1(self):
        self.intersect()
        self.area_intersect_set = interval_set_area(self.intersect_set)
        self.area_intersect_set[self.none_mask] = 0
        ans = self.area_intersect_set / self.interval_1_area
        return ans












