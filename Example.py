from SplitDataset_GenerateBbx import BbxGenerator
from Main_and_Analysis import Main
import pandas as pd

dataset_list = ['iris', 'breast_cancer', 'heloc', 'compas', 'shop']
dataset_name = 'iris'
Explanation_Task = BbxGenerator(dataset_name, max_depth=3, n_estimators=10)
Explanation_Task.generate(1)

di = Explanation_Task.dataset_information
train_idx = Explanation_Task.train_idx
explain_idx = Explanation_Task.explain_idx
test_idx = Explanation_Task.test_idx
rf = Explanation_Task.rf

APtree = Main(rf=rf,
              dataset_information=di,
              max_depth=5,
              explain_idx=explain_idx,
              test_idx=test_idx)
Result, prediction, _ = APtree.run()
Result_df = pd.DataFrame.from_dict(Result)
print(Result_df)
