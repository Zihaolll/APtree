import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import random
from Dataset_preprocess import get_dataset
import os
from sklearn.preprocessing import QuantileTransformer
from shutil import rmtree
def ml_file(file_path):
    # if os.path.exists(file_path):
        # rmtree(file_path)
    os.makedirs(file_path, exist_ok=True)

class BbxGenerator:
    def __init__(self, dataset_name, max_depth=3, n_estimators=5, random_seed=0, data_split_ratio=[0.6, 0.2, 0.2],
                 uni_switch=False, features_num=False):
        self.dataset_name = dataset_name
        self.data, self.x_columns, self.y_column = get_dataset(self.dataset_name)
        if features_num == False:
            self.data_X = self.data[self.x_columns].values
        else:
            self.data_X = self.data[self.x_columns[:features_num]].values
        self.data_Y = self.data[self.y_column].values
        if uni_switch == True:
            self.data_X = QuantileTransformer().fit_transform(self.data_X)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.init_random_seed = random_seed
        self.split_ratio = data_split_ratio
        self.data_upper_bound = np.max(self.data_X, axis=0)
        self.data_lower_bound = np.min(self.data_X, axis=0)
        self.space_lower_bound = self.data_lower_bound - 0.01 * (self.data_upper_bound - self.data_lower_bound)
        self.space_upper_bound = self.data_upper_bound + 0.01 * (self.data_upper_bound - self.data_lower_bound)
        random.seed(self.init_random_seed)
        self.class_name = np.unique(self.data_Y).tolist()
        self.Y_ratio = []
        for c_name in self.class_name:
            self.Y_ratio.append(((c_name == self.data_Y)+0).sum()/self.data_Y.shape[0])
        encoder = LabelEncoder()
        self.data_Y = encoder.fit_transform(self.data_Y)

        self.dataset_information = {'data_upper_bound': self.data_upper_bound,
                                    'data_lower_bound': self.data_lower_bound,
                                    'space_upper_bound': self.space_upper_bound,
                                    'space_lower_bound': self.space_lower_bound,
                                    'X': self.data_X,
                                    'Y': self.data_Y,
                                    'X_columns': self.x_columns,
                                    'Y_columns': self.y_column,
                                    'Y_name': self.class_name,
                                    'class_ratio': self.Y_ratio
                                    }
    def data_idx_split(self, seed):
        rand_state = self.init_random_seed + seed
        idx = np.arange(self.data.shape[0])
        train_ratio, val_ratio, test_ratio = self.split_ratio[0], self.split_ratio[1], self.split_ratio[2]
        _train_idx, _valandtest_idx = train_test_split(idx, train_size=train_ratio, shuffle=True,
                                                       random_state=rand_state, stratify=self.data_Y)
        _val_idx, _test_idx = train_test_split(_valandtest_idx, train_size=test_ratio / (1 - train_ratio), shuffle=True,
                                               random_state=rand_state)
        return _train_idx, _val_idx, _test_idx

    def rf_generator(self, train_idx):
        train_X = self.data_X[train_idx]
        train_Y = self.data_Y[train_idx]
        rf = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=1)
        rf.fit(train_X, train_Y)
        return rf

    def gbdt_generator(self, train_idx):
        train_X = self.data_X[train_idx]
        train_Y = self.data_Y[train_idx]
        GBDTf = GradientBoostingClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators, init='zero',
                                           learning_rate=0.1, loss='deviance', random_state=1)
        GBDTf.fit(train_X, train_Y)
        return GBDTf

    def save_bbx_task(self, path, train_idx, val_idx, test_idx, rf):
        np.save(os.path.join(path, 'train.npy'), train_idx)
        np.save(os.path.join(path, 'val.npy'), val_idx)
        np.save(os.path.join(path, 'test.npy'), test_idx)
        joblib.dump(rf, os.path.join(path, 'TreeEnsemble.pkl'))
        return

    def save_dataset_information(self, path):
        joblib.dump(self.dataset_information, os.path.join(path, 'dataset_information.pkl'))
        return
    def generate_rf(self, path):
        ml_file(os.path.join(path, self.dataset_name))
        self.save_dataset_information(os.path.join(path, self.dataset_name))
        train_idx, val_idx, test_idx = self.data_idx_split(0)
        rf = self.rf_generator(train_idx)
        save_path = os.path.join(path, self.dataset_name, 'RF')
        ml_file(save_path)
        self.save_bbx_task(save_path, train_idx, val_idx, test_idx, rf)
        return 'finish'
    def generate_gbdt(self, path):
        ml_file(os.path.join(path, self.dataset_name))
        self.save_dataset_information(os.path.join(path, self.dataset_name))
        train_idx, val_idx, test_idx = self.data_idx_split(0)
        gbdt = self.gbdt_generator(train_idx)
        save_path = os.path.join(path, self.dataset_name, 'GBDT')
        ml_file(save_path)
        self.save_bbx_task(save_path, train_idx, val_idx, test_idx, gbdt)
        return 'finish'


if __name__ == '__main__':
    dataset_name = 'electricity'
    path = 'Explanation_Task'
    BBX_G = BbxGenerator(dataset_name)
    BBX_G.generate_rf(path)
    BBX_G.generate_gbdt(path)
