"""
GNN Model
"""

# Libraries
import numpy as np
import pandas as pd
import torch
import Functions_PyG
import sklearn.metrics as metric
import Model_GNNs
import time
import os
import pickle
import random
import progressbar


file_name_train = 'Files/example_discovery.csv'
file_name_test = 'Files/example_validation.csv'

RANDOM_SEED = 121
time_start = time.time()

# Path: project/logging.txt
file_prefix = Functions_PyG.FilePrefix(file_name_train)
file_logging = 'logging_KNNGraph_' + file_prefix + '.txt'
if os.path.exists(os.getcwd() + '/' + file_logging):
    os.remove(os.getcwd() + '/' + file_logging)


# Data Preparation
data_train, features_train, labels_train, feature_number_train, sample_number_train = \
    Functions_PyG.DataReader(file_name_train, file_logging)
data_test, features_test, labels_test, feature_number_test, sample_number_test = \
    Functions_PyG.DataReader(file_name_test, file_logging)
class_labels, _ = Functions_PyG.LabelCounter(labels_train)
num_classes = len(class_labels)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.int64)

# File Dir (Path: project/subtype/data/)
plot_path = Functions_PyG.MakeNewDir(os.getcwd(), 'Plots/' + file_prefix, file_logging)
file_path = Functions_PyG.MakeNewDir(os.getcwd(), 'Results/' + file_prefix, file_logging)

# GNN Model (and Params)
CurrentGNNModel = Model_GNNs.ClusterGCNDNNModel
device = torch.device('cpu')

# Hyper Params (KNN Graph)
list_k_values = [5, 7, 9]  # full search of K
list_str_knn_type = ['both_neighbor']
list_str_distance = ['euclidean', 'cosine', 'pearson', 'spearman', 'RBFKernel', 'PolyKernel', 'SigKernel', 'LapKernel']
list_str_quality = ['EdgeNumDiff']

# Hyper Params (GNN Model)
list_lr = [0.001, 0.005, 0.01, 0.05, 0.1]
list_lz1 = [32, 64, 128] 
list_lz2 = [32, 64, 128] 
list_epoch_num = [20] 
MAX_EPOCH_NUM = list_epoch_num[-1]

# Result Summary
list_record_k_value, list_record_str_knn_type, list_record_str_distance, list_record_str_quality = [], [], [], []
list_record_lr, list_record_lz1, list_record_lz2, list_record_epoch_num = [], [], [], []
list_record_auc_train, list_record_auc_test = [], []

# KNN Graph Params Looping 
for k_value in list_k_values:
    for str_knn_type in list_str_knn_type:
        for str_distance in list_str_distance:
            for str_quality in list_str_quality:

                Functions_PyG.PrintAndSave(['========================================'], file_logging)
                Functions_PyG.PrintAndSave(['KNN Graph setting:', str(k_value), str_knn_type, str_distance, str_quality], file_logging)

                # Network Loading (Train Samples)
                object_file_prefix = os.getcwd() + '/Objects/' + str(k_value) + '_' + str_knn_type + '_' + str_distance + '_' + str_quality + '_'
                with open(object_file_prefix + file_prefix + '_train_edge_list.data', 'rb') as f_train:
                    pair_list_all_direction_train, edge_list_all_direction_train, data_train_scaled_selected = pickle.load(f_train)
                num_selected = data_train_scaled_selected.shape[1]  # num of selected features

                # PyG Data Transform (Train Samples)
                Functions_PyG.PrintAndSave(['Data Transform.'], file_logging)
                data_network_train = Functions_PyG.NetworkFromEdgeList_NodeSample(pair_list_all_direction_train,
                                                                                  edge_list_all_direction_train,
                                                                                  labels_train,
                                                                                  data_train_scaled_selected)
                data_network_train = data_network_train.to(device)  # data to device

                # GNN Model Params Looping 
                for learning_rate in list_lr:
                    for layer_size1 in list_lz1:
                        for layer_size2 in list_lz2:

                            Functions_PyG.PrintAndSave(['--------------------'], file_logging)
                            Functions_PyG.PrintAndSave(['.Params:', learning_rate, layer_size1, layer_size2], file_logging)
                            Functions_PyG.SetRandomSeed(RANDOM_SEED)

                            # Training Outer 
                            Functions_PyG.PrintAndSave(['.Training. (Outer Setting)'], file_logging)
                            gnn_model = CurrentGNNModel(int(num_selected), num_classes, int(layer_size1), int(layer_size2)).to(device)
                            optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)
                            optimizer_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                            loss_function = torch.nn.CrossEntropyLoss()

                            # Train Sample Info
                            train_auc_epochs_list = []
                            # Test Sample Info
                            test_auc_epochs_list = []
                            # Prediction Probability
                            train_prob_save, test_prob_save = [], []

                            # Epoch Looping ###############
                            Functions_PyG.PrintAndSave(['.Epoch Looping.'], file_logging)
                            widgets = ['Looping:', progressbar.Percentage(), ' ', progressbar.Bar('='), ' ', progressbar.Timer()]
                            pb_epoch = progressbar.ProgressBar(widgets=widgets, maxval=MAX_EPOCH_NUM).start()
                            for epoch in range(MAX_EPOCH_NUM):

                                # Forward ## Training
                                gnn_model.train()
                                output_train_epoch = gnn_model(data_network_train)

                                # Results ## Training
                                loss_train_epoch = loss_function(output_train_epoch, data_network_train.y)
                                _, predict_train_epoch = torch.max(output_train_epoch, 1)
                                train_prob_epoch = output_train_epoch[:, 1].detach().numpy().copy()
                                train_auc_epoch = metric.roc_auc_score(data_network_train.y.numpy(), train_prob_epoch)

                                # Backward ## Training
                                optimizer.zero_grad()
                                loss_train_epoch.backward()
                                optimizer.step()
                                optimizer_lr.step()

                                # Different Epochs (focus) ##############
                                # (Reduce the Calculations in Validation)
                                if (epoch + 1) in list_epoch_num:

                                    # Forward ## Test
                                    gnn_model.eval()
                                    preds_test_temp = np.zeros(sample_number_test, dtype=np.int64)
                                    probs_test_temp = np.zeros(sample_number_test, dtype=np.float32)
                                    output_test_epoch = torch.zeros((sample_number_test, num_classes), dtype=torch.float32)

                                    # Test One by One
                                    with torch.no_grad():
                                        for index_test_sample in range(sample_number_test):
                                            # Network Loading (Train Samples + One Sample)
                                            with open(object_file_prefix + file_prefix + '_test_edge_list_' + str(index_test_sample) + '.data', 'rb') as f_test:
                                                pair_list_all_direction_test_temp, edge_list_all_direction_test_temp, data_test_scaled_selected_temp = pickle.load(f_test)
                                            # PyG Data Transform (Train Samples + One Sample)
                                            data_network_test_temp = Functions_PyG.NetworkFromEdgeList_NodeSample(pair_list_all_direction_test_temp,
                                                                                                                  edge_list_all_direction_test_temp,
                                                                                                                  labels_test,
                                                                                                                  data_test_scaled_selected_temp)
                                            data_network_test_temp = data_network_test_temp.to(device)  # data to device

                                            # Result
                                            output_test_epoch_temp = gnn_model(data_network_test_temp)
                                            _, predict_test_epoch_temp = torch.max(output_test_epoch_temp, 1)
                                            output_test_epoch[index_test_sample, ] = output_test_epoch_temp[-1, ]
                                            preds_test_temp[index_test_sample] = predict_test_epoch_temp[-1].numpy()
                                            probs_test_temp[index_test_sample] = output_test_epoch_temp[-1, 1].detach().numpy()

                                    # Test Result (all test samples)
                                    loss_test_epoch = loss_function(output_test_epoch, labels_test_tensor)
                                    test_auc_epoch = metric.roc_auc_score(labels_test, probs_test_temp)
                                    test_prob_epoch = probs_test_temp.copy()

                                    Functions_PyG.PrintAndSave(['.Epoch.', epoch + 1], file_logging)

                                    # Results Record ## Training
                                    train_auc_epochs_list.append(train_auc_epoch)

                                    # Results Record ## Test
                                    test_auc_epochs_list.append(test_auc_epoch)
                                  

                                    # Prediction Prob Record  ## Settings
                                    list_record_k_value.append(str(k_value))
                                    list_record_str_knn_type.append(str_knn_type)
                                    list_record_str_distance.append(str_distance)
                                    list_record_str_quality.append(str_quality)
                                    list_record_lr.append(learning_rate)
                                    list_record_lz1.append(layer_size1)
                                    list_record_lz2.append(layer_size2)
                                    list_record_epoch_num.append(epoch + 1)
                                    list_record_auc_train.append(train_auc_epochs_list[-1])
                                    list_record_auc_test.append(test_auc_epochs_list[-1])

                                    # Target Prediction Probability
                                    train_prob_save = train_prob_epoch.copy()
                                    test_prob_save = test_prob_epoch.copy()

                                    df_train_prob_save = pd.DataFrame({'train_prob': train_prob_save})
                                    file_train_prob_save = file_path + '/' + str(k_value) + '_' + str_knn_type + '_' + str_distance + '_' + str_quality + '_'
                                    file_train_prob_save += str(learning_rate) + '_' + str(layer_size1) + '_' + str(layer_size2) + '_' + str(epoch + 1) + '_train_probs.csv'
                                    df_train_prob_save.to_csv(file_train_prob_save)

                                    df_test_prob_save = pd.DataFrame({'test_prob': test_prob_save})
                                    file_test_prob_save = file_path + '/' + str(k_value) + '_' + str_knn_type + '_' + str_distance + '_' + str_quality + '_'
                                    file_test_prob_save += str(learning_rate) + '_' + str(layer_size1) + '_' + str(layer_size2) + '_' + str(epoch + 1) + '_test_probs.csv'
                                    df_test_prob_save.to_csv(file_test_prob_save)

                                pb_epoch.update(epoch)
                            pb_epoch.finish()
                            # Training Outer 

                            # Time Mark
                            time_end_temp = time.time()
                            Functions_PyG.PrintAndSave(['Running', (time_end_temp - time_start) / 60, 'min'], file_logging)

df_Summary = pd.DataFrame({'record_str_k_value': list_record_k_value, 'record_str_knn_type': list_record_str_knn_type,
                           'record_str_distance': list_record_str_distance, 'record_str_quality': list_record_str_quality,
                           'record_lr': list_record_lr, 'record_lz1': list_record_lz1,
                           'record_lz2': list_record_lz2, 'record_epoch_num': list_record_epoch_num,
                           'record_auc_train': list_record_auc_train, 'record_auc_test': list_record_auc_test})
df_Summary.to_csv('result_summary_KNNGraph_' + file_prefix + '.csv')

# Result Summary
time_end = time.time()
Functions_PyG.PrintAndSave(['================================='], file_logging)
Functions_PyG.PrintAndSave(['GNN, Opt', 'KNN Graph'], file_logging)
Functions_PyG.PrintAndSave(['File: ', file_prefix], file_logging)
Functions_PyG.PrintAndSave(['Full Time: ', (time_end-time_start)/60, 'min'], file_logging)
Functions_PyG.PrintAndSave([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())], file_logging)
Functions_PyG.PrintAndSave(['================================='], file_logging)
