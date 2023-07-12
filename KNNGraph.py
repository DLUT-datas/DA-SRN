"""
Sample Network For Node Classification.
"""

# Libraries
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as skl_pairwise
import networkx as nx
from networkx.algorithms import community
import sys
import collections
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


# KNN Connections ###################################################


def KNNGraphFromAdjMatrix_Both(adj_matrix, k_value):
    """
    Generate KNN graph from adj matrix (Function 2).
    When occur in both neighbors, link these two nodes.

    :param adj_matrix: adj matrix (not dis_matrix, non-negative)
    :param k_value: k for KNN
    :return: adjusted matrix
    """
    num_nodes = adj_matrix.shape[0]
    adj_matrix_temp = adj_matrix.copy()  # copy
    # K value check
    if k_value > num_nodes - 1:
        print('K Cant Exceed Neighbor Number.')
        sys.exit(1)

    # keep the K neighbors (by set small values to 0)
    adj_matrix_zero_mask_row = np.zeros_like(adj_matrix, dtype=np.bool_)
    for i in range(num_nodes):
        node_weights = adj_matrix[i, ]
        small_values_index = np.argsort(node_weights)[range(int(num_nodes - k_value))]  # ascending
        adj_matrix_zero_mask_row[i, small_values_index] = True
    adj_matrix_zero_mask_col = adj_matrix_zero_mask_row.T
    adj_matrix_zero_mask = np.logical_or(adj_matrix_zero_mask_row, adj_matrix_zero_mask_col)  # logical or
    adj_matrix_temp[adj_matrix_zero_mask] = 0

    return adj_matrix_temp


# Pair-wise Distance ##############################################################


def AdjacencyMatrix(data_samples, str_distance):
    """
    Adjacency Matrix (General Function)

    :param data_samples: data, samples * features
    :param str_distance: string for distance in knn
    :return: distance matrix
    """
    if str_distance == 'euclidean':
        return AdjacencyMatrix_EuclideanDistance(data_samples)
    elif str_distance == 'cosine':
        return AdjacencyMatrix_CosineSimilarity(data_samples)
    elif str_distance == 'pearson':
        return AdjacencyMatrix_PearsonSimilarity(data_samples)
    elif str_distance == 'spearman':
        return AdjacencyMatrix_SpearmanSimilarity(data_samples)


def AdjacencyMatrix_EuclideanDistance(data_samples):
    """
    Adjacency Matrix (1 / Euclidean Distance)

    :param data_samples: data, samples * features
    :return: distance matrix
    """
    num_samples = data_samples.shape[0]
    dis_matrix = skl_pairwise.euclidean_distances(data_samples)
    adj_matrix = 1 / (dis_matrix + 1)
    adj_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    return adj_matrix


def AdjacencyMatrix_CosineSimilarity(data_samples):
    """
    Adjacency Matrix (abs(Cosine Similarity))

    :param data_samples: data, samples * features
    :return: adj_matrix [0, 1]
    """
    num_samples = data_samples.shape[0]
    corr_matrix = skl_pairwise.cosine_similarity(data_samples)
    corr_matrix[np.diag_indices(num_samples)] = 0
    adj_matrix = np.abs(corr_matrix)
    return adj_matrix


def AdjacencyMatrix_PearsonSimilarity(data_samples):
    """
    Adjacency Matrix (abs(Pearson Similarity))

    :param data_samples: data, samples * features
    :return: adj_matrix [0, 1]
    """
    # pearson correlation (samples)
    num_samples = data_samples.shape[0]
    data_temp = data_samples.T  # features * samples
    corr_matrix = np.array(pd.DataFrame(data_temp).corr(method='pearson'), dtype=np.float32)
    corr_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    adj_matrix = np.abs(corr_matrix)
    return adj_matrix


def AdjacencyMatrix_SpearmanSimilarity(data_samples):
    """
    Adjacency Matrix (abs(Spearman Similarity))

    :param data_samples: data, samples * features
    :return: adj_matrix [0, 1]
    """
    # spearman correlation (samples)
    num_samples = data_samples.shape[0]
    data_temp = data_samples.T  # features * samples
    corr_matrix = np.array(pd.DataFrame(data_temp).corr(method='spearman'), dtype=np.float32)
    corr_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    adj_matrix = np.abs(corr_matrix)
    return adj_matrix


# KNN Graph (Add one node)
def AdjacencyMatrix_AddNodes(source_matrix, all_matrix, k_value):
    """
    Add nodes to current adj matrix.
    Link the 'k_value' nodes to other nodes directly.
    Get the adj matrix with all nodes.

    :param source_matrix: original matrix (adj matrix)
    :param all_matrix: all node matrix (adj matrix)
    :param k_value: value k
    :return: new adj matrix
    """
    # number of nodes
    num_nodes_source = source_matrix.shape[0]  # number of source nodes
    num_nodes_all = all_matrix.shape[0]  # number of all nodes
    # new matrix prepare
    new_matrix = np.zeros_like(all_matrix, dtype=np.float32)
    new_matrix[0:num_nodes_source, 0:num_nodes_source] = source_matrix
    # add node one by one
    for i in range(num_nodes_source, num_nodes_all):
        # new node weights
        target_values = all_matrix[i, 0:i]
        large_values_index = np.argsort(target_values)[(k_value-1):]
        # keep 'k_value' neighbors
        filled_target_values = np.zeros_like(target_values, dtype=np.float32)
        filled_target_values[large_values_index] = target_values[large_values_index]
        # fill the new matrix
        new_matrix[i, 0:i] = filled_target_values
        new_matrix[0:i, i] = filled_target_values

    return new_matrix


# Graph Partition Quality ##############################################################

def PartitionQuality(adj_matrix, node_sets, str_quality):
    """
    Graph Partition Quality.

    :param adj_matrix: adjacency matrix
    :param node_sets: list of node sets (as partition)
    :param str_quality: string for quality
    :return: float
    """
    if str_quality == 'EdgeNumDiff':
        return GraphEdgeNumDiff(adj_matrix, node_sets)


def GraphEdgeNumDiff(adj_matrix, node_sets):
    """
    The number difference between intra-edges and inter-edges.

    :param adj_matrix: adjacency matrix
    :param node_sets: list of node sets (as partition)
    :return: int
    """
    graph = nx.from_numpy_matrix(adj_matrix)
    num_edges = nx.number_of_edges(graph)
    num_intra_edges = community.coverage(graph, node_sets) * num_edges  # intra edges
    num_inter_edges = num_edges - num_intra_edges  # inter edges
    EdgeNumDiff_value = num_intra_edges - num_inter_edges

    return EdgeNumDiff_value


def SampleNodeSets(labels):
    """
    Node sets (communities) defined on sample labels.
    The item in node sets is the node index. (0,1,2,3,...)

    :param labels: sample labels
    :return: list of node sets
    """
    class_labels, class_numbers = LabelCounter(labels)
    nodes_list = np.array(range(len(labels)), dtype=np.int64)
    mask_node_sets_labels = [labels == item for item in class_labels]
    node_sets = [nodes_list[mask] for mask in mask_node_sets_labels]

    return node_sets


def LabelCounter(labels):
    """
    Count the label array.

    :param labels: label array
    :return: class_labels, class_numbers
    """
    class_counter = collections.Counter(labels)
    class_dict = dict(sorted(class_counter.items()))
    class_labels = list(class_dict.keys())
    class_numbers = list(class_dict.values())

    return class_labels, class_numbers

