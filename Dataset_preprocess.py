import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer


''' 
read_data_XXX():output pd数据，数据特征索引，标签索引，特征情况

'''




def read_data_iris():
    iris = load_iris()
    data = pd.DataFrame(iris.data[:], columns=iris.feature_names)
    data['class'] = iris.target
    y_column = 'class'
    feature_types = ['float'] * 4
    x_columns = iris.feature_names

    return data, x_columns, y_column, feature_types


def read_data_breast_cancer_data():
    x_columns = ['Clump thickness',
                 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    y_column = 'class'
    feature_types = ['float'] * 9
    data = pd.read_csv('DataSet/breast-cancer-wisconsin.data', names=x_columns+[y_column])
    data = data[data['Bare Nuclei'] != '?']
    data['Bare Nuclei'] = [int(i) for i in data['Bare Nuclei']]
    return data, x_columns, y_column, feature_types


def read_data_heloc_uni():
    data = pd.read_csv('DataSet/heloc')
    x_columns = [col for col in data.columns[1:-1]]
    y_column = 'class'
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types

def read_data_heloc():
    data = pd.read_csv('DataSet/heloc_preprocessed.csv')
    x_columns = [col for col in data.columns[2:-1]]
    y_column = data.columns[1]
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types

def read_data_shop():
    data = pd.read_csv('DataSet/shop')
    x_columns = [col for col in data.columns[1:-1]]
    y_column = 'class'
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types

def read_data_compas():
    data = pd.read_csv('DataSet/compas')
    x_columns = [col for col in data.columns[1:-1]]
    y_column = 'class'
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types

def read_data_compas_uni():
    data = pd.read_csv('DataSet/compas')
    x_columns = [col for col in data.columns[1:-1]]
    y_column = 'class'
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types



def get_dataset(dataset_name):
    if dataset_name == 'iris':
        return read_data_iris()
    elif dataset_name == 'breast_cancer':
        return read_data_breast_cancer_data()
    elif dataset_name == 'compas':
        return read_data_compas()
    elif dataset_name == 'shop':
        return read_data_shop()
    elif dataset_name == 'heloc':
        return read_data_heloc()
    elif dataset_name == 'heloc_uni':
        return read_data_heloc_uni()




