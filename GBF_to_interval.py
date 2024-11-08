import numpy as np



def interval_updata_left(current_interval, threshold):
    if current_interval[0] < threshold:
        current_interval[1] = threshold
    return current_interval


def interval_updata_right(current_interval, threshold):
    if current_interval[1] > threshold:
        current_interval[0] = threshold
    return current_interval


def _parse_GBF_interval(model, left_bounds, right_bounds):
    n_features = model.n_features_in_
    leaf_interval = []
    leaf_pridict = []

    for step_indx in range(model.estimators_.shape[0]):
        for class_indx in range(model.estimators_.shape[1]):
            tree = model.estimators_[step_indx, class_indx]
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            values = tree.tree_.value

            nodes = np.ones((n_nodes, n_features, 2))
            nodes[:, :, 0] = nodes[:, :, 0] * left_bounds
            nodes[:, :, 1] = nodes[:, :, 1] * right_bounds

            for i in range(n_nodes):

                if (children_left[i] != children_right[i]):
                    nodes[children_left[i]] = nodes[i].copy()
                    nodes[children_right[i]] = nodes[i].copy()  # 继承父节点区间
                    current_interval = nodes[i][feature[i]]
                    nodes[children_left[i]][feature[i]] = interval_updata_left(current_interval.copy(), threshold[i])
                    nodes[children_right[i]][feature[i]] = interval_updata_right(current_interval.copy(),
                                                                                 threshold[i])  # 区间更新

                else:
                    if model.n_classes_ <= 2:
                        pridict_values = model.learning_rate * values[i][0]

                    else:
                        pridict_values = np.zeros(shape=[model.n_classes_])
                        pridict_values[class_indx] = model.learning_rate * values[i][0]

                    leaf_pridict.append(pridict_values)
                    leaf_interval.append(nodes[i])
    leaf_interval_set, leaf_proba_set = np.array(leaf_interval), np.array(leaf_pridict)
    return leaf_interval_set, leaf_proba_set


def _parse_GBFeachTree_interval_with(model, left_bounds, right_bounds):
    n_features = model.n_features_in_
    GBF_interval = []
    GBF_proba = []

    for step_indx in range(model.estimators_.shape[0]):
        for class_indx in range(model.estimators_.shape[1]):
            leaf_interval = []
            leaf_pridict = []
            tree = model.estimators_[step_indx, class_indx]
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            values = tree.tree_.value

            nodes = np.ones((n_nodes, n_features, 2))
            nodes[:, :, 0] = nodes[:, :, 0] * left_bounds
            nodes[:, :, 1] = nodes[:, :, 1] * right_bounds

            for i in range(n_nodes):

                if children_left[i] != children_right[i]:
                    nodes[children_left[i]] = nodes[i].copy()
                    nodes[children_right[i]] = nodes[i].copy()  # 继承父节点区间
                    current_interval = nodes[i][feature[i]]
                    nodes[children_left[i]][feature[i]] = interval_updata_left(current_interval.copy(), threshold[i])
                    nodes[children_right[i]][feature[i]] = interval_updata_right(current_interval.copy(),
                                                                                 threshold[i])  # 区间更新

                else:
                    if model.n_classes_ <= 2:
                        pridict_values = model.learning_rate * values[i][0]

                    else:
                        pridict_values = np.zeros(shape=[model.n_classes_])
                        pridict_values[class_indx] = model.learning_rate * values[i][0]
                    leaf_pridict.append(pridict_values)
                    leaf_interval.append(nodes[i])
            leaf_interval_set, leaf_proba_set = np.array(leaf_interval), np.array(leaf_pridict)


            GBF_interval.append(leaf_interval_set)
            GBF_proba.append(leaf_proba_set)

    return GBF_interval, GBF_proba
