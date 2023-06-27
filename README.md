# APtree
Explaining Random Forests as Single Decision Trees

## An Example
### Creating an Explanation Task
```python
from SplitDataset_GenerateBbx import BbxGenerator
from Main_and_Analysis import Main
import pandas as pd

dataset_name = 'iris'
Explanation_Task = BbxGenerator(dataset_name, max_depth=3, n_estimators=10)
Explanation_Task.generate()
```
An Explanation Task includes: Dataset and original RF.

The dataset is divided into three subsets with the ratio of $(6:2:2)$.
These subsets are used for training the original RF, constructing the APtee and Evaluating the APtree.

The `dataset_name` parameter can be set with `'iris', 'breast_cancer', 'heloc', 'compas', 'shop'`.


The `max_depth` and `n_estimators` parameters are used for generate the original RF.


### Reading the Explanation Task
```python
di = Explanation_Task.dataset_information
explain_idx = Explanation_Task.explain_idx
test_idx = Explanation_Task.test_idx
rf = Explanation_Task.rf
```
The original `rf` has been trained with the traning set. 

### Setting, Constructing and Evaluating APtree
```
APtree = Main(rf=rf,
         dataset_information=di,
         max_depth=5,
         explain_idx=explain_idx，
         test_idx=test_idx)
Result, prediction, _ = APtree.run()
Result_df = pd.DataFrame.from_dict(Result)
```
The `Main` function includes the Evaluating process.

Parameters of `Main`:
- `max_depth` is the $d_{max}$ in Algorithm 1 (APtree-Generator).

The remaining parameters are optional:
- `alpha` is the $\alpha$ in Algorithm 1 (APtree-Generator). Defaults to 0.
- `vector_distance` is the $\rho$ of the distance functional, either`'NL'`or`'CL'`. Defaults to `'NL'`.

