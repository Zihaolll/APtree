from SplitDataset_GenerateBbx import BbxGenerator
from Main_and_Analysis import Main
import pandas as pd

##  第一步构建一个解释任务：包含已经数据集信息及各个子数据集索引以及原始森林模型

dataset_list = ['iris', 'breast_cancer', 'heloc', 'compas', 'shop']
dataset_name = 'compas'
Explanation_Task = BbxGenerator(dataset_name, max_depth=3, n_estimators=10)
Explanation_Task.generate(1)

##  提取模型
di = Explanation_Task.dataset_information
train_idx = Explanation_Task.train_idx
val_idx = Explanation_Task.val_idx
test_idx = Explanation_Task.test_idx
rf = Explanation_Task.rf

##  初始化解释问题原森林模型rf, 数据信息，最大深度参数max_depth对应与论文中dmax, alpha=0是早停条件，
M = Main(rf=rf,
         dataset_information=di,
         max_depth=5,
         alpha=0,
         SP_size=-1,
         test_idx=test_idx,
         val_idx=val_idx,
         measure='Distribution',
         vector_distance='NL')

##  构造解释
Result, prediction, _ = M.run()
Result_df = pd.DataFrame.from_dict(Result)


