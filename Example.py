import joblib
import os
import numpy as np
from APtree import APtree

path = 'Explanation_Task'
dataset_name = 'shop'
load_path_information = os.path.join(path, dataset_name, 'dataset_information.pkl')
dataset_information = joblib.load(load_path_information)
EnsembleMechanism = 'RF'

load_path_bbx = os.path.join(path, dataset_name, EnsembleMechanism)
train_idx = np.load(os.path.join(load_path_bbx, 'train.npy'), allow_pickle=True)
val_idx = np.load(os.path.join(load_path_bbx, 'val.npy'), allow_pickle=True)
test_idx = np.load(os.path.join(load_path_bbx, 'test.npy'), allow_pickle=True)
TE = joblib.load(os.path.join(load_path_bbx, 'TreeEnsemble.pkl'))


Explanation = APtree(TreeEnsemble=TE, dataset_information=dataset_information, max_step=5, test_idx=test_idx,
                     val_idx=val_idx, measure='Distribution', vector_distance='NL')

ANS, _, _ = Explanation.run()
print(ANS)
