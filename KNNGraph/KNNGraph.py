"""
Sample Network For Node Classification.
"""

# Libraries
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as skl_pairwise
import minepy
import networkx as nx
from networkx.algorithms import community
import Functions_KNNGraph
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


# KNN Connections ###################################################

def KNNGraphFromAdjMatrix(adj_matrix, k_value, str_knn_type):
    """
    Generate KNN graph from adj matrix.
    (According to str_knn_type)

    :param adj_matrix: adj matrix (not dis_matrix, non-negative)
    :param k_value: k for KNN
    :param str_knn_type: type for connections, {'any_neighbor', 'both_neighbor'}
    :return: adjusted matrix
    """
    if str_knn_type == 'any_neighbor':
        return KNNGraphFromAdjMatrix_Any(adj_matrix, k_value)
    elif str_knn_type == 'both_neighbor':
        return KNNGraphFromAdjMatrix_Both(adj_matrix, k_value)


def KNNGraphFromAdjMatrix_Any(adj_matrix, k_value):
    """
    Generate KNN graph from adj matrix.
    When occur in any neighbors, link these two nodes.

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
    adj_matrix_zero_mask = np.logical_and(adj_matrix_zero_mask_row, adj_matrix_zero_mask_col)  # logical and
    adj_matrix_temp[adj_matrix_zero_mask] = 0

    return adj_matrix_temp


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
    elif str_distance == 'RBFKernel':
        return AdjacencyMatrix_RBFKernel(data_samples)
    elif str_distance == 'PolyKernel':
        return AdjacencyMatrix_PolyKernel(data_samples)
    elif str_distance == 'SigKernel':
        return AdjacencyMatrix_SigKernel(data_samples)
    elif str_distance == 'LapKernel':
        return AdjacencyMatrix_LapKernel(data_samples)
    elif str_distance == 'ChiKernel':
        return AdjacencyMatrix_ChiKernel(data_samples)


def AdjacencyMatrix_EuclideanDistance(data_samples):
    """
    Adjacency Matrix 

    :param data_samples: data, samples * features
    :return: distance matrix
    """
    num_samples = data_samples.shape[0]
    dis_matrix = skl_pairwise.euclidean_distances(data_samples)
    adj_matrix = 1 / (dis_matrix + 1)
    adj_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    return adj_matrix


def AdjacencyMatrix_ManhattanDistance(data_samples):
    """
    Adjacency Matrix 

    :param data_samples: data, samples * features
    :return: distance matrix
    """
    num_samples = data_samples.shape[0]
    dis_matrix = skl_pairwise.manhattan_distances(data_samples)
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


def AdjacencyMatrix_MICSimilarity(data_samples):
    """
    Adjacency Matrix (MIC)

    :param data_samples: data, samples * features
    :return: adj_matrix [0, 1]
    """
    # maximum information coefficient
    num_samples = data_samples.shape[0]
    corr_matrix = minepy.cstats(X=data_samples, Y=data_samples)[0]
    corr_matrix = np.array(corr_matrix, dtype=np.float32)
    corr_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    adj_matrix = corr_matrix  # for uniform format
    return adj_matrix


# Kernel Functions

def AdjacencyMatrix_RBFKernel(data_samples):
    """
    Adjacency Matrix (RBF Kernel)

    :param data_samples: data, samples * features
    :return: adj_matrix （0, 1）
    """
    # kernel for similarity
    num_samples = data_samples.shape[0]
    sim_matrix = skl_pairwise.rbf_kernel(data_samples)
    sim_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    adj_matrix = sim_matrix
    return adj_matrix


def AdjacencyMatrix_PolyKernel(data_samples):
    """
    Adjacency Matrix (Poly Kernel)

    :param data_samples: data, samples * features
    :return: adj_matrix
    """
    # kernel for similarity
    num_samples = data_samples.shape[0]
    sim_matrix = skl_pairwise.polynomial_kernel(data_samples)
    sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))  # normalization
    sim_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    adj_matrix = sim_matrix
    return adj_matrix


def AdjacencyMatrix_SigKernel(data_samples):
    """
    Adjacency Matrix (Sigmoid Kernel)

    :param data_samples: data, samples * features
    :return: adj_matrix
    """
    # kernel for similarity
    num_samples = data_samples.shape[0]
    sim_matrix = skl_pairwise.sigmoid_kernel(data_samples)
    sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))  # normalization
    sim_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    adj_matrix = sim_matrix
    return adj_matrix


def AdjacencyMatrix_LapKernel(data_samples):
    """
    Adjacency Matrix (Laplacian Kernel)

    :param data_samples: data, samples * features
    :return: adj_matrix
    """
    # kernel for similarity
    num_samples = data_samples.shape[0]
    sim_matrix = skl_pairwise.laplacian_kernel(data_samples)
    sim_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    adj_matrix = sim_matrix
    return adj_matrix


def AdjacencyMatrix_ChiKernel(data_samples):
    """
    Adjacency Matrix (Chi-square Kernel)
    Note: Data must be non-negative, thus not used actually.

    :param data_samples: data, samples * features
    :return: adj_matrix
    """
    # kernel for similarity
    num_samples = data_samples.shape[0]
    sim_matrix = skl_pairwise.chi2_kernel(data_samples)
    sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))  # normalization
    sim_matrix[np.diag_indices(num_samples)] = 0  # set diagonal 0
    adj_matrix = sim_matrix
    return adj_matrix


# Functions

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
    if str_quality == 'modularity':
        return GraphModularity(adj_matrix, node_sets)
    elif str_quality == 'coverage':
        return GraphCoverage(adj_matrix, node_sets)
    elif str_quality == 'performance':
        return GraphPerformance(adj_matrix, node_sets)
    elif str_quality == 'connections':
        return GraphConnections(adj_matrix, node_sets)
    elif str_quality == 'EdgeNumDiff':
        return GraphEdgeNumDiff(adj_matrix, node_sets)


def GraphModularity(adj_matrix, node_sets):
    """
    Modularity Value. (for a partition)

    :param adj_matrix: adjacency matrix
    :param node_sets: list of node sets (as partition)
    :return: float
    """
    graph = nx.from_numpy_matrix(adj_matrix)
    modularity_value = community.modularity(graph, node_sets)

    return modularity_value


def GraphCoverage(adj_matrix, node_sets):
    """
    Coverage Value. (for a partition)

    :param adj_matrix: adjacency matrix
    :param node_sets: list of node sets (as partition)
    :return: float
    """
    graph = nx.from_numpy_matrix(adj_matrix)
    coverage_value = community.coverage(graph, node_sets)

    return coverage_value


def GraphPerformance(adj_matrix, node_sets):
    """
    Performance Value. (for a partition)

    :param adj_matrix: adjacency matrix
    :param node_sets: list of node sets (as partition)
    :return: float
    """
    graph = nx.from_numpy_matrix(adj_matrix)
    performance_value = community.performance(graph, node_sets)

    return performance_value


def GraphConnections(adj_matrix, node_sets):
    """
    Performance Value. (for a partition)
    Equals the number of potential minus the number of inter edges.

    :param adj_matrix: adjacency matrix
    :param node_sets: list of node sets (as partition)
    :return: float
    """
    graph = nx.from_numpy_matrix(adj_matrix)
    num_nodes = nx.number_of_nodes(graph)
    num_potential_edges = num_nodes * (num_nodes - 1) / 2
    num_inter_edges = (1 - community.coverage(graph, node_sets)) * nx.number_of_edges(graph)
    connectivity_value = num_potential_edges - num_inter_edges

    return connectivity_value


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


# Other Functions ######

def ShowPartitionQuality(adj_matrix, node_sets):
    """
    Print Graph Partition Quality.

    :param adj_matrix: adjacency matrix
    :param node_sets: list of node sets (as partition)
    :return: float values
    """
    value_modularity = GraphModularity(adj_matrix, node_sets)
    value_coverage = GraphCoverage(adj_matrix, node_sets)
    value_performance = GraphPerformance(adj_matrix, node_sets)
    value_connections = GraphConnections(adj_matrix, node_sets)
    value_EdgeNumDiff = GraphEdgeNumDiff(adj_matrix, node_sets)

    return value_modularity, value_coverage, value_performance, value_connections, value_EdgeNumDiff


def ShowGraphBasicInfo(adj_matrix, node_sets):
    """
    Basic info of graph.

    :param adj_matrix:
    :param node_sets: list of node sets (as partition)
    :return:
    """
    graph = nx.from_numpy_matrix(adj_matrix)
    num_nodes = nx.number_of_nodes(graph)  # number of nodes
    num_edges = nx.number_of_edges(graph)  # number of edges
    value_density = nx.density(graph)  # density

    num_intra_edges = community.coverage(graph, node_sets) * num_edges  # intra edges
    num_inter_edges = num_edges - num_intra_edges  # inter edges

    return num_nodes, num_edges, value_density, num_intra_edges, num_inter_edges


def SampleNodeSets(labels):
    """
    Node sets (communities) defined on sample labels.
    The item in node sets is the node index. (0,1,2,3,...)

    :param labels: sample labels
    :return: list of node sets
    """
    class_labels, class_numbers = Functions_KNNGraph.LabelCounter(labels)
    nodes_list = np.array(range(len(labels)), dtype=np.int64)
    mask_node_sets_labels = [labels == item for item in class_labels]
    node_sets = [nodes_list[mask] for mask in mask_node_sets_labels]

    return node_sets


def KNNGraphShow(adj_matrix, node_labels, save_path):
    """
    Show the graph.

    :param adj_matrix: adj matrix
    :param node_labels: labels for nodes
    :param save_path: figure save path
    :return: figure
    """
    graph = nx.from_numpy_matrix(adj_matrix)
    color_candidates = ['red', 'dodgerblue', 'orange', 'gold', 'limegreen']
    color_list = [color_candidates[label] for label in node_labels]
    fig = plt.figure(figsize=(18, 12))
    nx.draw_networkx(graph, with_labels=False, node_color=color_list, pos=nx.kamada_kawai_layout(graph))
    ax_off = plt.axis("off")
    plt.savefig(save_path, dpi=300)
    plt.close()
