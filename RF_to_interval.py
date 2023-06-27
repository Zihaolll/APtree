import numpy as np
# import cupy as np

def interval_updata_left(current_interval, threshold):
    if current_interval[0] < threshold:
        current_interval[1] = threshold
    return current_interval


def interval_updata_right(current_interval, threshold):
    if current_interval[1] > threshold:
        current_interval[0] = threshold
    return current_interval


def _parse_forest_interval(model, left_bounds, right_bounds):
    n_estimators = model.n_estimators
    n_classes = len(model.classes_)
    n_features = model.n_features_in_
    leaf_interval = []
    leaf_pridict_prob = []

    for j in range(n_estimators):
        tree = model.estimators_[j]
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
                nodes[children_right[i]] = nodes[i].copy()
                current_interval = nodes[i][feature[i]]
                nodes[children_left[i]][feature[i]] = interval_updata_left(current_interval.copy(), threshold[i])
                nodes[children_right[i]][feature[i]] = interval_updata_right(current_interval.copy(),
                                                                             threshold[i])

            else:
                leaf_pridict_prob.append(values[i] / values[i].sum())
                leaf_interval.append(nodes[i])

    leaf_interval_set, leaf_proba_set = np.array(leaf_interval), np.array(leaf_pridict_prob)
    leaf_proba_set = leaf_proba_set.reshape((leaf_proba_set.shape[0], leaf_proba_set.shape[2]))
    return leaf_interval_set, leaf_proba_set

def _parse_forest_interval_with_flux(model, left_bounds, right_bounds):
    n_estimators = model.n_estimators
    n_classes = len(model.classes_)
    n_features = model.n_features_in_
    leaf_interval = []
    leaf_pridict_prob = []
    leaf_flux = []
    for j in range(n_estimators):
        tree = model.estimators_[j]
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
                nodes[children_right[i]] = nodes[i].copy()
                current_interval = nodes[i][feature[i]]
                nodes[children_left[i]][feature[i]] = interval_updata_left(current_interval.copy(), threshold[i])
                nodes[children_right[i]][feature[i]] = interval_updata_right(current_interval.copy(),
                                                                             threshold[i])

            else:
                leaf_pridict_prob.append(values[i] / values[i].sum())
                leaf_interval.append(nodes[i])
                leaf_flux.append(values[i].sum())
    leaf_interval_set, leaf_proba_set, leaf_flux_set = np.array(leaf_interval), np.array(leaf_pridict_prob), np.array(leaf_flux)

    leaf_proba_set = leaf_proba_set.reshape((leaf_proba_set.shape[0], leaf_proba_set.shape[2]))

    return leaf_interval_set, leaf_proba_set, leaf_flux_set



def _parse_tree_interval_with_flux(model, left_bounds, right_bounds):

    n_classes = len(model.classes_)
    n_features = model.n_features_in_
    leaf_interval = []
    leaf_pridict_prob = []
    leaf_flux = []
    for j in range(1):
        tree = model
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
                nodes[children_right[i]] = nodes[i].copy()
                current_interval = nodes[i][feature[i]]
                nodes[children_left[i]][feature[i]] = interval_updata_left(current_interval.copy(), threshold[i])
                nodes[children_right[i]][feature[i]] = interval_updata_right(current_interval.copy(),
                                                                             threshold[i])

            else:
                leaf_pridict_prob.append(values[i] / values[i].sum())
                leaf_interval.append(nodes[i])
                leaf_flux.append(values[i].sum())
    leaf_interval_set, leaf_proba_set, leaf_flux_set = np.array(leaf_interval), np.array(leaf_pridict_prob), np.array(
        leaf_flux)

    leaf_proba_set = leaf_proba_set.reshape((leaf_proba_set.shape[0], leaf_proba_set.shape[2]))

    return leaf_interval_set, leaf_proba_set, leaf_flux_set


def _parse_RFeachTree_interval_with_flux(model, left_bounds, right_bounds):

    n_classes = len(model.classes_)
    n_features = model.n_features_in_
    RF_interval = []
    RF_proba = []
    RF_flux = []
    n_estimators = model.n_estimators

    for j in range(n_estimators):
        leaf_interval = []
        leaf_pridict_prob = []
        leaf_flux = []

        tree = model.estimators_[j]
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
                leaf_pridict_prob.append(values[i] / values[i].sum())
                leaf_interval.append(nodes[i])
                leaf_flux.append(values[i].sum())
        leaf_interval_set, leaf_proba_set, leaf_flux_set = np.array(leaf_interval), np.array(leaf_pridict_prob), np.array(
            leaf_flux)

        leaf_proba_set = leaf_proba_set.reshape((leaf_proba_set.shape[0], leaf_proba_set.shape[2]))

        RF_interval.append(leaf_interval_set)
        RF_proba.append(leaf_proba_set)
        RF_flux.append(leaf_flux_set)
    return RF_interval, RF_proba, RF_flux

