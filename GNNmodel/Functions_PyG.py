"""
Common functions.
"""

# Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric import data
from torch_geometric.transforms import OneHotDegree  # noqa
import torch
import torch.nn.functional as F  # noqa
import collections
import os
import shutil
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def SetRandomSeed(RANDOM_SEED):
    """
    Set random seed.

    :param RANDOM_SEED: seed value
    :return: none
    """
    random.seed(RANDOM_SEED)  # module random: seed
    np.random.seed(RANDOM_SEED)  # module numpy: seed
    torch.manual_seed(RANDOM_SEED)  # module torch: seed
    torch.cuda.manual_seed_all(RANDOM_SEED)  # module torch: cuda seed


def DataReader(file_name, file_logging):
    """
    Read the data from file.

    :param file_name: data file path
    :param file_logging: info save path
    :return: Data, Features, Labels, FeatureNumber, SampleNumber
    """
    file_data = pd.read_csv(file_name, header=None)  # read data file

    features_read = np.array(file_data.iloc[1:, 0], dtype=np.int64)  # vector of features, int64
    labels_read = np.array(file_data.iloc[0, 1:], dtype=np.int64)  # vector of labels, int64
    data_read = np.array(file_data.iloc[1:, 1:], dtype=np.float32).T  # matrix of values, samples * features, float32

    feature_number = len(features_read)  # number of features
    sample_number = len(labels_read)  # number of samples

    PrintAndSave(['--- Dataset Summary ---'], file_logging)
    PrintAndSave(['File:', file_name.split('/')[1]], file_logging)
    PrintAndSave(['Sample Number:', sample_number], file_logging)
    PrintAndSave(['Feature Number:', feature_number], file_logging)

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


def PairListMapping(pair_list):
    """
    Map the node of pair list to new index (0,1,2,3,...).
    Origin node ids may have special meaning, for example, (2,3,4,7,...)
    New node ids are unified as index (0,1,2,3,...).

    :param pair_list: original pair_list
    :return: pair_list_new
    """
    nodes = np.union1d(pair_list[:, 0], pair_list[:, 1])  # original index, auto unique
    new_index = np.array(range(len(nodes)), dtype=np.int64)  # new index: 0,1,2,3,...
    dict_index = dict(zip(nodes, new_index))  # dict
    pair_list_new = np.vectorize(dict_index.get)(pair_list)

    return pair_list_new


def AdjacencyMatrixToEdgeList(adj_matrix):
    """
    Transform adjacency matrix to pair list and edge list.
    Adjacency matrix contains all connections based on all non-zero values.
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
    nodes = np.union1d(pair_list_all_direction[:, 0], pair_list_all_direction[:, 1])

    return pair_list_all_direction, edge_list_all_direction, nodes


def NetworkFromEdgeList_NodeSample(pair_list_all_direction, edge_list_all_direction, node_labels, node_attr_matrix):
    """
    Generate PyG Data object from edge list.
    This function is designed for a node-level task. 'y' is the node labels.
    Note: inductive task, only contains the current labels.

    :param pair_list_all_direction: pair list
    :param edge_list_all_direction: edge list
    :param node_labels: node (sample) labels
    :param node_attr_matrix: node attr matrix
    :return: PyG object
    """
    # pair list
    pair_list_all_direction_new = PairListMapping(pair_list_all_direction)
    num_pair = pair_list_all_direction_new.shape[0]
    # edge_index
    edge_index = torch.tensor(pair_list_all_direction_new, dtype=torch.long)
    # edge_attr
    edge_attr = torch.tensor(edge_list_all_direction[:, 0].reshape((num_pair, 1)), dtype=torch.float32)

    # y
    y = torch.tensor(node_labels, dtype=torch.int64)  # node (sample) label
    # x
    x = torch.tensor(node_attr_matrix, dtype=torch.float32)  # node * attr

    # Data object
    data_object = data.Data(edge_index=edge_index.t().contiguous(),
                            edge_attr=edge_attr,
                            y=y,
                            x=x)

    return data_object


def DegreeListFromPairList(pair_list, target_features, is_directed):
    """
    Get degree list for target_features in pair_list.
    Undirected graph have no in_degree, out_degree.

    :param pair_list: list of pairs (array, two columns, node1, node2) (0,1,2,3,...)
    :param target_features: target features, may have no order (array) (0,1,2,3,...)
    :param is_directed: True/False (bool)
    :return: all_degrees, out_degrees, in_degrees (tensors)
    """
    all_degrees = torch.tensor([np.sum(pair_list == f) for f in target_features], dtype=torch.int64)
    out_degrees = torch.zeros(len(target_features), dtype=torch.int64)
    in_degrees = torch.zeros(len(target_features), dtype=torch.int64)
    # directed: node1 -> node2
    if is_directed:
        out_degrees = torch.tensor([np.sum(pair_list[:, 0] == f) for f in target_features], dtype=torch.int64)
        in_degrees = torch.tensor([np.sum(pair_list[:, 1] == f) for f in target_features], dtype=torch.int64)

    return all_degrees, out_degrees, in_degrees


def NodeFeaturesFromOneHotDegree(degree_list):
    """
    Get the node feature matrix by on-hot-degree coding.

    :param degree_list: degree of nodes (tensor)
    :return: x (node feature matrix, tensor, float)
    """
    # one hot code, default code length is maximum + 1
    x = F.one_hot(degree_list).to(torch.float32)

    return x


def NodeFeaturesFromIdentityMatrix(index_exist_nodes, num_features):
    """
    Get the node feature matrix by identity matrix.

    :param index_exist_nodes: index of graph nodes
    :param num_features: node numbers (int)
    :return: x (node feature matrix, tensor, float)
    """
    # identity matrix according to node number
    x = torch.eye(num_features, dtype=torch.float32)
    x = x[index_exist_nodes, :]

    return x


def MetricsPlot(save_path, num_epochs, dict_metrics_list, dict_loss_list):
    """
    Plot for acc and loss in epoch loop.

    :param save_path: plot save path.
    :param num_epochs: epoch number
    :param dict_metrics_list: dict of metric list (keys: names, values: metrics)
    :param dict_loss_list: list of loss
    :return: no return
    """
    x_range = np.arange(1, num_epochs+1)
    fig, ax1 = plt.subplots()  # a plot with an axis

    # basic of metric list
    list_metric_names = list(dict_metrics_list.keys())
    list_loss_names = list(dict_loss_list.keys())
    list_colors = ['orange', 'royalblue', 'limegreen', 'violet', 'red',
                   'deeppink', 'gold', 'deepskyblue', 'yellowgreen']

    # Draw lines of metrics
    list_lines = []
    index = 0
    for str_metric in list_metric_names:
        temp_line, = ax1.plot(x_range, dict_metrics_list[str_metric], color=list_colors[index])
        list_lines.append(temp_line)
        index += 1
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Metrics')

    # Draw lines of 'loss'
    ax2 = ax1.twinx()  # a new axis
    for str_metric in list_loss_names:
        temp_line, = ax2.plot(x_range, dict_loss_list[str_metric], color=list_colors[index])
        list_lines.append(temp_line)
        index += 1
    ax2.set_ylabel('Loss')

    # Draw the Legend
    list_names = list_metric_names + list_loss_names
    ax2.legend(list_lines, list_names)

    # Save the pic
    plt.savefig(save_path)
    plt.close()


def PairListToCSV(save_path, pair_list, edge_list, file_logging):
    """
    Save network(pair_list, edge_list) into csv file.

    :param save_path: file path
    :param pair_list: pair list (node1, node2)
    :param edge_list: edge list (weight, type)
    :param file_logging: info save path
    :return: no return
    """
    pair_df = pd.DataFrame(np.column_stack((pair_list, edge_list)),
                           columns=['node1', 'node2', 'weight', 'type'])
    pair_df['node1'] = pair_list[:, 0]
    pair_df['node2'] = pair_list[:, 1]
    pair_df.to_csv(save_path)
    PrintAndSave(['Pair List Saved.'], file_logging)


def MakeNewDir(target_dir, file_prefix, file_logging):
    """
    Make a dir = target_dir + file_prefix

    :param target_dir: target folder
    :param file_prefix: name of new folder
    :param file_logging: info save path
    :return: new folder path
    """
    full_path = target_dir + '/' + file_prefix
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
        PrintAndSave([file_prefix, 'Exists. Remove.'], file_logging)
    os.makedirs(full_path)
    PrintAndSave([file_prefix, 'Made.'], file_logging)

    return full_path


def PrintAndSave(list_str, save_path):
    """
    Print and save at the same time.
    First combine the str in list.

    :param list_str: the string to output
    :param save_path: the file name (path)
    :return: no return
    """
    # list of string
    str_info = list_str[0]
    for s in range(1, len(list_str)):
        str_info = str_info + ' ' + str(list_str[s])
    # output to the Console
    print(str_info)
    # output to the File
    with open(save_path, "a+") as f_write:
        print(str_info, file=f_write)
