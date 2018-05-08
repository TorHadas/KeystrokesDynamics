from NetworkClass import Network
import os
import numpy as np
from plotResults import plot

folder_dir = os.getcwd() + "\\logs"

# network constants
DEFAULT_ITER_NUM = 200
DAMP = 0.001
NETWORK_SIZE = 3
LAYER_SIZE = [
    25,  # input
    11,  # hidden
    5  # output - 11 users, 0 non-user (default)
]
ITER_NUM = 200

network = Network(DAMP, ITER_NUM, NETWORK_SIZE, LAYER_SIZE)
network.get_data(folder_dir)
network.train_network()

def identification_decider(result):
    maximum = 0
    for i, value in enumerate(result):
        if value > maximum:
            maximum = value
            index = i
    result_answer = np.zeros(len(result))
    result_answer[index] = 1
    return result_answer


def test(network, data):
    imposter = 1; othentic = 0;
    true_identifications = 0
    false_identification = 0
    false_rejection_rate = 0
    false_acceptance_rate = 0
    mean_cost_vec = []
    max_cost_vec = []
    for test_element in data:
        net_result = network.run(test_element[1])
        mean_cost_vec.append((sum(network.cost(net_result, test_element[0])) / len(test_element)) ** (1/2))
        max_cost_vec.append(max(network.cost(net_result, test_element[0])) ** (1/2))
        net_decision = identification_decider(net_result)
        if net_decision == test_element[0]:
            true_identifications += 1
        else:
            false_identification += 1
            if test_element[0][othentic] == 1:
                false_rejection_rate += 1
            else:
                false_acceptance_rate += 1

    iter_vec = np.arange(len(mean_cost_vec))
    x_data = [iter_vec] * 2
    y_data = [mean_cost_vec, max_cost_vec]
    plot(x_data, y_data, "sample num", "costs", "bla")
    false_rejection_rate = false_rejection_rate / len(data)
    false_acceptance_rate = false_acceptance_rate / len(data)
    true_identifications_pct = true_identifications / len(data)
    print("frr is: " + str(false_rejection_rate) +  " far is: " + str(false_acceptance_rate) +
          " good: " + str(true_identifications_pct))