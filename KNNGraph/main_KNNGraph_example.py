"""
KNN Model for Feature Selection.
Keep the connected network for each test sample.
Connect the test sample to the training network directly.
"""

# Libraries
import numpy as np
import pandas as pd
import Functions_KNNGraph
import GASelection_KNNSpecial_proportion_KSearch
import KNNGraph
import sklearn.metrics as metric
import time
import progressbar
import os
import pickle


file_name_train = 'Files/example_discovery.csv'
file_name_test = 'Files/example_validation.csv'

RANDOM_SEED = 121
time_start = time.time()


# Hyper Params (KNN Graph)
list_k_values = [5, 7, 9]  
list_str_knn_type = ['both_neighbor']
# list_str_distance： ['euclidean', 'cosine', 'pearson', 'spearman']
# list_str_distance = ['RBFKernel', 'PolyKernel', 'SigKernel', 'LapKernel', 'ChiKernel']
list_str_distance = ['cosine']
list_str_quality = ['EdgeNumDiff']

# Data Preparation
file_prefix = Functions_KNNGraph.FilePrefix(file_name_train)
data_train, features_train, labels_train, feature_number_train, sample_number_train = \
    Functions_KNNGraph.DataReader(file_name_train)
data_test, features_test, labels_test, feature_number_test, sample_number_test = \
    Functions_KNNGraph.DataReader(file_name_test)
class_labels, _ = Functions_KNNGraph.LabelCounter(labels_train)
num_classes = len(class_labels)

# Data Scaling
data_train_scaled, data_test_scaled = Functions_KNNGraph.UVScaling(data_train, data_test)  # scaling

# File Dir
plot_files_path = Functions_KNNGraph.MakeNewDir(os.getcwd(), '/Plots')  # folder of plots
object_files_path = Functions_KNNGraph.MakeNewDir(os.getcwd(), '/Objects')  # folder of objects
feature_files_path = Functions_KNNGraph.MakeNewDir(os.getcwd(), '/Features')  # folder of features

# Store
length_settings = len(list_k_values) * len(list_str_knn_type) * len(list_str_distance) * len(list_str_quality)
list_k_values_setting, list_knn_type_setting, list_distance_setting, list_quality_setting = [], [], [], []

col_names_df_train = ['modularity_train', 'coverage_train', 'performance_train', 'connections_train', 'EdgeNumDiff_train']
df_metrics_train = pd.DataFrame(np.zeros((length_settings, 5), dtype=np.float32), columns=col_names_df_train)
df_basic_info_train = pd.DataFrame(np.zeros((length_settings, 5), dtype=np.float32),
                                   columns=['num_nodes_train', 'num_edges_train', 'density_train', 'num_intra_edges_train', 'num_inter_edges_train'])

# Great Loop
index_settings = 0  # index of settings
for k_value in list_k_values:
    for str_knn_type in list_str_knn_type:
        for str_distance in list_str_distance:
            for str_quality in list_str_quality:
                print('---------------------------------------')
                print('Hyper Params', k_value, str_knn_type, str_distance, str_quality)
                np.random.seed(RANDOM_SEED)
                time_start_temp = time.time()

                list_k_values_setting.append(str(k_value))
                list_knn_type_setting.append(str_knn_type)
                list_distance_setting.append(str_distance)
                list_quality_setting.append(str_quality)

                # Network Construction
                print('KNNGraph Selection')
                mask_selected = GASelection_KNNSpecial_proportion_KSearch.GASelectionForKNNGraph(num_features=feature_number_train,
                                                                                                 size_pop=100,
                                                                                                 proportion_ones=0.1,
                                                                                                 k_value=k_value,
                                                                                                 max_iter=100,
                                                                                                 num_elites=1,
                                                                                                 ratio_mutation=1/feature_number_train,
                                                                                                 ratio_crossover=0.5,
                                                                                                 data_train=data_train_scaled,
                                                                                                 labels_train=labels_train,
                                                                                                 str_distance=str_distance,
                                                                                                 str_knn_type=str_knn_type,
                                                                                                 str_quality=str_quality)

                # Selected Summary
                num_selected = np.sum(mask_selected)
                print('num_selected', num_selected)
                feature_file_prefix = feature_files_path + '/' + str(k_value) + '_' + str_knn_type + '_' + str_distance + '_' + str_quality + '_'
                df_features = pd.DataFrame({"selected": mask_selected})
                df_features.to_csv(feature_file_prefix + file_prefix + '_features_' + str(num_selected) + '.csv')

                # Data for Selected
                data_train_scaled_selected = data_train_scaled[:, mask_selected].reshape((sample_number_train, num_selected))
                data_test_scaled_selected = data_test_scaled[:, mask_selected].reshape((sample_number_test, num_selected))

                ###########################
                # KNN Graph (Train Samples)
                adj_matrix_train = KNNGraph.AdjacencyMatrix(data_train_scaled_selected, str_distance)
                adj_matrix_train_knn = KNNGraph.KNNGraphFromAdjMatrix(adj_matrix_train, k_value, str_knn_type)

                # KNN Graph Basic Info (Train Samples)
                num_nodes_train, num_edges_train, value_density_train, num_intra_edges_train, num_inter_edges_train = \
                    KNNGraph.ShowGraphBasicInfo(adj_matrix_train_knn, KNNGraph.SampleNodeSets(labels_train))
                df_basic_info_train.iloc[index_settings, ] = num_nodes_train, num_edges_train, value_density_train, num_intra_edges_train, num_inter_edges_train
                print('Basic (train):', 'nodes', num_nodes_train, 'edges', num_edges_train, 'density', value_density_train)
                print('Basic (train):', 'intra-edges', num_intra_edges_train, 'inter-edges', num_inter_edges_train)

                # KNN Partition Quality (Train Samples)
                value_modularity_train, value_coverage_train, value_performance_train, value_connections_train, value_EdgeNumDiff_train = \
                    KNNGraph.ShowPartitionQuality(adj_matrix_train_knn, KNNGraph.SampleNodeSets(labels_train))
                df_metrics_train.iloc[index_settings, ] = value_modularity_train, value_coverage_train, value_performance_train, value_connections_train, value_EdgeNumDiff_train
                print('Quality (train):', 'modularity', value_modularity_train, 'coverage', value_coverage_train)
                print('Quality (train):', 'performance', value_performance_train, 'connections', value_connections_train)
                print('Quality (train):', 'EdgeNumDiff', value_EdgeNumDiff_train)

                # Save the Plots (Train Samples)
                plot_file_prefix = plot_files_path + '/' + str(k_value) + '_' + str_knn_type + '_' + str_distance + '_' + str_quality + '_'
                KNNGraph.KNNGraphShow(adj_matrix_train_knn, labels_train, plot_file_prefix + file_prefix + 'knn_graph_train.png')

                # Matrix to Edge List (Train Samples)
                pair_list_all_direction_train, edge_list_all_direction_train, nodes_train = Functions_KNNGraph.AdjacencyMatrixToEdgeList(adj_matrix_train_knn)

                # Object Saving (Train Samples)
                print('Object Save. (Train Samples)')
                object_file_prefix = object_files_path + '/' + str(k_value) + '_' + str_knn_type + '_' + str_distance + '_' + str_quality + '_'
                with open(object_file_prefix + file_prefix + '_train_edge_list.data', 'wb') as f_train:
                    pickle.dump([pair_list_all_direction_train, edge_list_all_direction_train, data_train_scaled_selected], f_train)
                ###########################

                # KNN Graph (Train Samples + Test Sample)
                widgets = ['Test Samples:', progressbar.Percentage(), ' ', progressbar.Bar('='), ' ', progressbar.Timer()]
                pb_test_samples = progressbar.ProgressBar(widgets=widgets, maxval=sample_number_test).start()
                for index_test_sample in range(sample_number_test):

                    # train samples + one test sample
                    data_test_scaled_selected_one_test_sample = np.vstack((data_train_scaled_selected, data_test_scaled_selected[index_test_sample, ]))

                    # KNN Graph (Train Samples + One Sample)
                    adj_matrix_test = KNNGraph.AdjacencyMatrix(data_test_scaled_selected_one_test_sample, str_distance)

                    # KNN Graph (Train Samples + One Sample) (Special: direct links)
                    adj_matrix_test_knn_new = KNNGraph.AdjacencyMatrix_AddNodes(adj_matrix_train_knn, adj_matrix_test, k_value)

                    # Matrix to Edge List (Train Samples + One Sample)
                    pair_list_all_direction_test, edge_list_all_direction_test, nodes_test = Functions_KNNGraph.AdjacencyMatrixToEdgeList(adj_matrix_test_knn_new)

                    # Save the Objects (Train Samples + One Sample)
                    with open(object_file_prefix + file_prefix + '_test_edge_list_' + str(index_test_sample) + '.data', 'wb') as f_test:
                        pickle.dump([pair_list_all_direction_test, edge_list_all_direction_test, data_test_scaled_selected_one_test_sample], f_test)

                    pb_test_samples.update(index_test_sample + 1)
                pb_test_samples.finish()

                # loop summary
                time_end_temp = time.time()
                print('loop duration:', (time_end_temp-time_start_temp)/60, 'min')
                index_settings = index_settings + 1

# Save the Settings and Metrics
df_settings = pd.DataFrame({'k_value': list_k_values_setting, 'knn_type': list_knn_type_setting, 'distance': list_distance_setting, 'quality': list_quality_setting})
df_quality = pd.concat([df_settings, df_metrics_train], axis=1)
df_quality.to_csv('KNNGraph_' + file_prefix + '_' + list_str_distance[0] + '_' + list_str_quality[0] + '_settings.csv')

# Save the Basic Info
df_Basic = pd.concat([df_settings, df_basic_info_train], axis=1)
df_Basic.to_csv('KNNGraph_' + file_prefix + '_' + list_str_distance[0] + '_' + list_str_quality[0] + '_basic.csv')

# Result Summary
time_end = time.time()
print('=================================')
print('Find KNN Graph (proportion - one test sample)')
print(list_str_distance)
print(list_str_quality)
print('File:', file_prefix)
print('Exe Time:', (time_end-time_start)/60, 'min')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print('=================================')
