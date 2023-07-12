"""
Genetic Algorithms for Feature Selection
"""

# Libraries
import numpy as np
import progressbar


def GetBestChromosome(pop, fitness_values, num_selected=1):
    """
    Get the best chromosome.

    :param pop: population
    :param fitness_values: fitness values
    :param num_selected: number for target
    :return: index (array)
    """
    cho_sizes = np.sum(pop, axis=1)
    index_pop_sort = np.lexsort((cho_sizes, -fitness_values))  # values in last column is primary key
    index_best = index_pop_sort[0:num_selected]
    index_others = index_pop_sort[num_selected:]

    return index_best, index_others



def InitialPopulationProportion(num_features, size_pop, proportion_ones):
    """
    Initial Population (Proportion).
    Set the proportion of 'one' in cho to control the feature number.

    :param num_features: number of features
    :param size_pop: number of chromosomes
    :param proportion_ones: proportion of 'one'
    :return: pop
    """
    num_ones = np.floor(num_features * proportion_ones)
    pop = np.zeros((size_pop, num_features), dtype=np.bool_)
    for i in range(size_pop):
        loc_bits = np.random.choice(num_features, size=int(num_ones), replace=True)
        pop[i, loc_bits] = True

    return pop


def SelectionOperationTournament(pop, fitness_values, size_need, num_athletes=2):
    """
    Selection (Tournament)

    :param pop: population
    :param fitness_values: fitness values
    :param size_need: number of need
    :param num_athletes: number of athletes
    :return: new population
    """
    size_pop = len(fitness_values)
    index_selected = np.zeros(size_need, dtype=np.int64)
    for i in range(size_need):
        index_athletes = np.random.choice(size_pop, size=num_athletes, replace=False)
        index_winner, _ = GetBestChromosome(pop[index_athletes, ], fitness_values[index_athletes], 1)
        index_selected[i] = index_athletes[index_winner][0]
    pop_new = pop[index_selected, ]

    return pop_new


def UniformCrossoverOperation(pop, ratio_crossover=0.5):
    """
    Uniform Crossover.

    :param pop: population
    :param ratio_crossover: ratio of crossover
    :return: new population
    """
    size_pop = pop.shape[0]
    num_features = pop.shape[1]
    # define random pairs (round down)
    index_array = np.random.choice(size_pop, size=np.where(size_pop % 2 == 0, size_pop, size_pop - 1), replace=False)
    num_pairs = int(size_pop / 2)
    index_crossover = np.array(index_array, dtype=np.int64).reshape((num_pairs, 2))
    pop_new = pop.copy()  # copy
    for i in range(num_pairs):
        if np.random.rand() < ratio_crossover:
            mask_crossover = np.random.rand(num_features) >= 0.5
            pop_new[index_crossover[i, 0], mask_crossover] = pop[index_crossover[i, 1], mask_crossover]
            pop_new[index_crossover[i, 1], mask_crossover] = pop[index_crossover[i, 0], mask_crossover]

    return pop_new


def BasicBitMutationOperation(pop, ratio_mutation):
    """
    Basic Bit Mutation

    :param pop: population
    :param ratio_mutation: ratio of mutation
    :return: new population
    """
    size_pop = pop.shape[0]
    num_features = pop.shape[1]
    # mutation
    pop_new = pop.copy()
    prob_mutation = np.random.rand(size_pop, num_features)
    mask_mutation = prob_mutation < ratio_mutation
    pop_new[mask_mutation] = ~pop[mask_mutation]

    return pop_new


def FitnessEvaluation(pop, data_train, labels_train, eval_fun):
    """
    Get Fitness Evaluation for One Chromosome

    :param pop: population
    :param data_train: data for fitness
    :param labels_train: label for fitness
    :param eval_fun: evaluation function
    :return: fitness values
    """
    size_pop = pop.shape[0]
    # get fitness values
    fitness_values = np.zeros(size_pop, dtype=np.float32)
    for i in range(size_pop):
        fitness_values[i] = eval_fun(data_train, labels_train, pop[i, ])

    return fitness_values

