"""
Common functions.
"""

# Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import collections
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def DataReader(file_name):
    """
    Read the data from file.

    :param file_name: data file path
    :return: Data, Features, Labels, FeatureNumber, SampleNumber
    """
    file_data = pd.read_csv(file_name, header=None)  # read data file

    features_read = np.array(file_data.iloc[1:, 0], dtype=np.int64)  # vector of features, int64
    labels_read = np.array(file_data.iloc[0, 1:], dtype=np.int64)  # vector of labels, int64
    data_read = np.array(file_data.iloc[1:, 1:], dtype=np.float32).T  # matrix of values, samples * features, float32

    feature_number = len(features_read)  # number of features
    sample_number = len(labels_read)  # number of samples

    print('--- Dataset Summary ---')
    print('File: ', file_name.split('/')[1])
    print('Sample Number: ', sample_number)
    print('Feature Number: ', feature_number)

    return data_read, features_read, labels_read, feature_number, sample_number


def FilePrefix(file_name):
    """
    Return the prefix of file.
    ('Files/XXX.csv')

    :param file_name: file path
    :return: prefix
    """
    prefix = file_name.split('/')[1].split('.')[0]

    return prefix


def LabelCounter(labels):
    """
    Count the label array.

    :param labels: label array
    :return: class_labels, class_numbers
    """
    class_counter = collections.Counter(labels)  # Counter object, dict
    class_dict = dict(sorted(class_counter.items()))  # sorted dict, ascending, 0,1,...
    class_labels = list(class_dict.keys())  # classes, list
    class_numbers = list(class_dict.values())  # number of classes, list

    return class_labels, class_numbers


def UVScaling(data_train, data_test):
    """
    Scaling by mean and std.

    :param data_train: training data, sample * feature
    :param data_test: test data, sample * feature
    :return: data_train_scaled, data_test_scaled
    """
    Scaler = StandardScaler()
    Scaler.mean_ = np.mean(data_train, axis=0)  # mean in training data
    Scaler.scale_ = np.std(data_train, axis=0, ddof=1)  # std in training data
    data_train_scaled = Scaler.transform(data_train)  # scaled training data
    data_test_scaled = Scaler.transform(data_test)  # scaled test data

    return data_train_scaled, data_test_scaled


def AdjacencyMatrixToEdgeList(adj_matrix):
    """
    Transform adjacency matrix to pair list and edge list.
    Adjacency matrix contains all connections based on all non-zero values. Thus ot is 'all_direction'.
    Node ids in pair list start from 0.

    :param adj_matrix: adjacency matrix
    :return: pair_list, edge_list, nodes
    """
    # extract non-zero values
    nonzero_index = np.nonzero(adj_matrix)  # non-zero index
    pair_list_all_direction = np.array(nonzero_index).T  # pair list
    edge_weight = adj_matrix[nonzero_index]  # non-zero values
    edge_type = np.where(edge_weight > 0, 1, -1)  # edge type
    edge_list_all_direction = np.column_stack((edge_weight, edge_type))
    # all used nodes (with > 1 edge) (same as index, 0,1,2...)
    nodes = np.union1d(pair_list_all_direction[:, 0], pair_list_all_direction[:, 1])  # no need

    return pair_list_all_direction, edge_list_all_direction, nodes


def NetworkToCSV(save_path, pair_list, edge_list):
    """
    Save network(pair_list, edge_list) into csv file.

    :param save_path: file path
    :param pair_list: pair list (node1, node2)
    :param edge_list: edge list (weight, type)
    :return:
    """
    pair_df = pd.DataFrame(np.column_stack((pair_list, edge_list)),
                           columns=['node1', 'node2', 'weight', 'type'])
    pair_df['node1'] = pair_list[:, 0]
    pair_df['node2'] = pair_list[:, 1]
    pair_df.to_csv(save_path)


def MakeNewDir(target_dir, file_prefix):
    """
    Make a dir = target_dir + file_prefix

    :param target_dir: target folder
    :param file_prefix: name of new folder
    :return: new folder path
    """
    full_path = target_dir + '/' + file_prefix
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
        print(full_path, 'exists. Remove.')
    os.makedirs(full_path)
    print(full_path, 'Made.')

    return full_path
