import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import random
from Dataset_preprocess import get_dataset
import os
from shutil import rmtree
from sklearn.preprocessing import MinMaxScaler


def ml_file(file_path):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


class BbxGenerator:
    def __init__(self, dataset_name, max_depth=5, n_estimators=50, random_seed=0, data_split_ratio=[0.6, 0.2, 0.2],
                 uni_switch=False, features_num=False):
        self.dataset_name = dataset_name
        self.data, self.x_columns, self.y_column, self.feature_types = get_dataset(self.dataset_name)
        if features_num == False:
            self.data_X = self.data[self.x_columns].values
        else:
            self.data_X = self.data[self.x_columns[:features_num]].values
        self.data_Y = self.data[self.y_column].values
        if uni_switch == True:
            self.data_X = MinMaxScaler().fit_transform(self.data_X)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.init_random_seed = random_seed
        self.split_ratio = data_split_ratio
        self.data_upper_bound = np.max(self.data_X, axis=0)
        self.data_lower_bound = np.min(self.data_X, axis=0)
        self.space_lower_bound = self.data_lower_bound - 0.01 * (self.data_upper_bound - self.data_lower_bound)
        self.space_upper_bound = self.data_upper_bound + 0.01 * (self.data_upper_bound - self.data_lower_bound)
        random.seed(self.init_random_seed)
        self.dataset_information = {'data_upper_bound': self.data_upper_bound,
                                    'data_lower_bound': self.data_lower_bound,
                                    'space_upper_bound': self.space_upper_bound,
                                    'space_lower_bound': self.space_lower_bound,
                                    'X': self.data_X,
                                    'Y': self.data_Y,
                                    'X_columns': self.x_columns,
                                    'Y_columns': self.y_column}

    def data_idx_split(self, seed):
        rand_state = self.init_random_seed + seed
        idx = np.arange(self.data.shape[0])
        train_ratio, val_ratio, test_ratio = self.split_ratio[0], self.split_ratio[1], self.split_ratio[2]
        _train_idx, _valandtest_idx = train_test_split(idx, train_size=train_ratio, shuffle=True,
                                                       random_state=rand_state)
        _val_idx, _test_idx = train_test_split(_valandtest_idx, train_size=test_ratio / (1 - train_ratio), shuffle=True,
                                               random_state=rand_state)
        return _train_idx, _val_idx, _test_idx

    def rf_generator(self, train_idx):
        train_X = self.data_X[train_idx]
        train_Y = self.data_Y[train_idx]
        rf = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=1)
        rf.fit(train_X, train_Y)
        return rf

    def save_bbx_task(self, path, train_idx, val_idx, test_idx, rf):
        np.save(os.path.join(path, 'train.npy'), train_idx)
        np.save(os.path.join(path, 'val.npy'), val_idx)
        np.save(os.path.join(path, 'test.npy'), test_idx)
        joblib.dump(rf, os.path.join(path, 'rf.pkl'))
        return

    def save_dataset_information(self, path):
        joblib.dump(self.dataset_information, os.path.join(path, 'dataset_information.pkl'))
        return

    def generate(self, seed=0):
        self.train_idx, self.val_idx, self.test_idx = self.data_idx_split(seed)
        self.rf = self.rf_generator(self.train_idx)
        return 'finish'


