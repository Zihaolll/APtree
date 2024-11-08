import pandas as pd
from sklearn.datasets import load_iris
from scipy.io import arff


def read_data_iris():
    iris = load_iris()
    data = pd.DataFrame(iris.data[:], columns=iris.feature_names)
    data['class'] = iris.target
    y_column = 'class'
    x_columns = iris.feature_names

    return data, x_columns, y_column


def read_data_heloc():
    data = pd.read_csv('DataSet/heloc_preprocessed.csv')
    x_columns = [col for col in data.columns[2:-1]]
    y_column = data.columns[1]
    return data, x_columns, y_column


def read_data_shop():
    data = pd.read_csv('DataSet/shop')
    x_columns = [col for col in data.columns[1:-1]]
    y_column = 'class'
    return data, x_columns, y_column


def read_data_compas():
    data = pd.read_csv('DataSet/compas')
    x_columns = [col for col in data.columns[1:-1]]
    y_column = 'class'
    return data, x_columns, y_column


def read_data_electricity():
    dataset, meta = arff.loadarff('DataSet/electricity.arff')
    df = pd.DataFrame(dataset)
    x_columns = ['date', 'period', 'nswprice', 'nswdemand', 'vicprice',
                 'vicdemand', 'transfer']
    y_column = 'class'
    class0 = df['class'][0]
    df['class'] = [0 if i == class0 else 1 for i in df['class']]
    data = df[x_columns + [y_column]]

    return data, x_columns, y_column


def get_dataset(dataset_name):
    if dataset_name == 'iris':
        return read_data_iris()
    elif dataset_name == 'compas':
        return read_data_compas()
    elif dataset_name == 'shop':
        return read_data_shop()
    elif dataset_name == 'heloc':
        return read_data_heloc()
    elif dataset_name == 'electricity':
        return read_data_electricity()
