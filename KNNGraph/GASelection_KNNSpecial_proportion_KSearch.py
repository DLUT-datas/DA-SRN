"""
Genetic Algorithms for Feature Selection
Special: KNN Graph, within full search of 'K'
"""

# Libraries
import numpy as np
import progressbar
import GASelection_proportion
import KNNGraph


def GASelectionForKNNGraph(num_features,
                           size_pop,
                           proportion_ones,
                           k_value,
                           max_iter,
                           num_elites,
                           ratio_mutation,
                           ratio_crossover,
                           data_train,
                           labels_train,
                           str_distance,
                           str_knn_type,
                           str_quality):
    """
    Simple GA for Feature Selection.

    :param num_features: number of features
    :param size_pop: size of population
    :param proportion_ones: proportion of 'ones'
    :param k_value: K for KNN Graph
    :param max_iter: max of iterations
    :param num_elites: number of elites (always 1)
    :param ratio_mutation: ratio of mutation
    :param ratio_crossover: ratio of crossover
    :param data_train: data for fitness
    :param labels_train: labels for fitness
    :param str_distance: string for indicating the distance in knn
    :param str_knn_type: string for indicating the knn type
    :param str_quality: string for quality type
    :return: mask of selected
    """
    # Initial Population
    print('-------------------')
    print('Simple GA')
    print('Initialization.')
    pop_features = InitialPopulationProportion_KNNGraphFeatures(num_features, size_pop, proportion_ones)

    print('Evaluation.')
    fitness_values = FitnessEvaluationPartitionQuality(pop_features, data_train, labels_train, k_value,
                                                       str_distance, str_knn_type, str_quality)
    index_elites, index_others = GASelection_proportion.GetBestChromosome(pop_features, fitness_values, num_elites)
    cho_global_best = pop_features[index_elites, ][0]
    fitness_global_best = fitness_values[index_elites][0]

    # Generation
    print('Generation.')
    widgets = ['Generation:', progressbar.Percentage(), ' ',  progressbar.Bar('='), ' ', progressbar.Timer()]
    pb = progressbar.ProgressBar(widgets=widgets, maxval=max_iter).start()
    for i in range(max_iter):

        # Selection
        pop_features_general = SelectionOperationTournament_KNNGraphFeatures(pop_features,
                                                                             fitness_values,
                                                                             size_pop,
                                                                             num_athletes=2)

        # Crossover
        pop_features_general = UniformCrossoverOperation_KNNGraphFeatures(pop_features_general,
                                                                          ratio_crossover)

        # Mutation
        pop_features_general = BasicBitMutationOperation_KNNGraphFeatures(pop_features_general,
                                                                          ratio_mutation)

        # Elitism Model
        fitness_values = FitnessEvaluationPartitionQuality(pop_features_general,
                                                           data_train,
                                                           labels_train,
                                                           k_value,
                                                           str_distance,
                                                           str_knn_type,
                                                           str_quality)

        index_elites, index_others = GASelection_proportion.GetBestChromosome(pop_features_general, fitness_values, num_elites)
        if fitness_values[index_elites][0] > fitness_global_best:
            cho_global_best = pop_features_general[index_elites, ][0]
            fitness_global_best = fitness_values[index_elites][0]

        # New Generation (with Elitism Strategy)
        pop_features = pop_features_general.copy()
        pop_features[index_others[-1], ] = cho_global_best
        fitness_values[index_others[-1]] = fitness_global_best

        pb.update(i+1)
    pb.finish()

    # Final Selection
    index_best_solution, _ = GASelection_proportion.GetBestChromosome(pop_features, fitness_values, 1)
    mask_selected_features = pop_features[index_best_solution, ][0]

    return mask_selected_features


def InitialPopulationRandom_KNNGraphFeatures(num_features, size_pop):
    """
    Initial Population for Features (Random).

    :param num_features: number of features
    :param size_pop: number of chromosomes
    :return: pop_features
    """
    # population for features
    pop_features = GASelection_proportion.InitialPopulationRandom(num_features, size_pop)

    return pop_features


def InitialPopulationProportion_KNNGraphFeatures(num_features, size_pop, proportion_ones):
    """
    Initial Population for Features (Proportion).

    :param num_features: number of features
    :param size_pop: number of chromosomes
    :param proportion_ones: proportion of 'one'
    :return: pop_features
    """
    # population for features
    pop_features = GASelection_proportion.InitialPopulationProportion(num_features, size_pop, proportion_ones)

    return pop_features


def UniformCrossoverOperation_KNNGraphFeatures(pop_features, ratio_crossover=0.5):
    """
    Uniform Crossover.

    :param pop_features: population for features
    :param ratio_crossover: ratio of crossover
    :return: new population
    """
    size_pop = pop_features.shape[0]
    num_features = pop_features.shape[1]
    # define random pairs (round down)
    index_array = np.random.choice(size_pop, size=np.where(size_pop % 2 == 0, size_pop, size_pop - 1), replace=False)
    num_pairs = int(size_pop / 2)
    index_crossover = np.array(index_array, dtype=np.int64).reshape((num_pairs, 2))
    # new populations
    pop_features_new = pop_features.copy()
    for i in range(num_pairs):
        if np.random.rand() < ratio_crossover:
            # crossover for features
            mask_crossover = np.random.rand(num_features) >= 0.5
            pop_features_new[index_crossover[i, 0], mask_crossover] = pop_features[index_crossover[i, 1], mask_crossover]
            pop_features_new[index_crossover[i, 1], mask_crossover] = pop_features[index_crossover[i, 0], mask_crossover]

    return pop_features_new


def BasicBitMutationOperation_KNNGraphFeatures(pop_features, ratio_mutation):
    """
    Basic Bit Mutation.

    :param pop_features: population for features
    :param ratio_mutation: ratio of mutation
    :return: new populations
    """
    size_pop = pop_features.shape[0]
    num_features = pop_features.shape[1]
    # new populations
    pop_features_new = pop_features.copy()
    # mutation for features
    prob_mutation_features = np.random.rand(size_pop, num_features)
    mask_mutation_features = prob_mutation_features < ratio_mutation
    pop_features_new[mask_mutation_features] = ~pop_features[mask_mutation_features]

    return pop_features_new


def SelectionOperationTournament_KNNGraphFeatures(pop_features, fitness_values, size_need, num_athletes=2):
    """
    Selection (Tournament).

    :param pop_features: population for features
    :param fitness_values: fitness values
    :param size_need: number of need
    :param num_athletes: number of athletes
    :return: new populations
    """
    size_pop = len(fitness_values)
    index_selected = np.zeros(size_need, dtype=np.int64)
    for i in range(size_need):
        index_athletes = np.random.choice(size_pop, size=num_athletes, replace=False)
        index_winner, _ = GASelection_proportion.GetBestChromosome(pop_features[index_athletes, ], fitness_values[index_athletes], 1)
        index_selected[i] = index_athletes[index_winner][0]
    pop_features_new = pop_features[index_selected, ]

    return pop_features_new


def FitnessEvaluationPartitionQuality(pop_features, data_train, labels_train, k_value,
                                      str_distance, str_knn_type, str_quality):
    """
    Get Fitness Evaluation for One Chromosome.
    Fitness is defined as graph modularity.

    :param pop_features: population for features
    :param data_train: data for fitness
    :param labels_train: labels for fitness
    :param k_value: K for KNN Graph
    :param str_distance: string for indicating the dis in knn graph
    :param str_knn_type: string for indicating the knn type
    :param str_quality: string for quality type
    :return: fitness values
    """
    size_pop = pop_features.shape[0]
    num_samples = len(labels_train)

    # get partition based on labels
    node_sets = KNNGraph.SampleNodeSets(labels_train)

    # get fitness values
    fitness_values = np.zeros(size_pop, dtype=np.float32)
    for i in range(size_pop):
        # features
        num_selected = sum(pop_features[i, :])
        data_train_selected = data_train[:, pop_features[i, :]].reshape((num_samples, num_selected))
        # adj matrix
        adj_matrix = KNNGraph.AdjacencyMatrix(data_train_selected, str_distance)
        # knn graph
        knn_graph_adj_matrix = KNNGraph.KNNGraphFromAdjMatrix(adj_matrix, k_value, str_knn_type)

        # fitness values
        fitness_values[i] = KNNGraph.PartitionQuality(knn_graph_adj_matrix, node_sets, str_quality)

    return fitness_values

