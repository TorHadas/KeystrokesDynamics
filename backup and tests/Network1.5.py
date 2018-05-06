import numpy as np
from Data import *
from plotResults import plot
import os
#folder_dir = "C:\\Users\\T8497069\\Desktop\\Smop\\KeystrokesDynamics\\data"
folder_dir = os.getcwd() + "\\logs"


np.set_printoptions(suppress=True,precision=5)
def activation_func(x, derivative=False):
    if derivative:
        y = np.nan_to_num(np.exp(-x))
        return np.nan_to_num(y * np.power((1 + y), 2))
    return 1 / (1 + np.exp(-x))


def cost(net_result, answer, derivative=False):
    if derivative:
        return -2 * np.power(answer - net_result , 1)
    return np.abs(np.power(net_result - answer, 2))


def train_network(folder_data_dir):

    #region: set network constants
    DAMP = 0.01
    NETWORK_SIZE = 2
    L = NETWORK_SIZE - 1
    LAYER_SIZE = [
        1,  # input
        1,  # hidden
        1,    # output
        1,    # output
        1,    # output
        2,    # output
        2    # output
    ]
    MAX_LAYER_SIZE = np.max(LAYER_SIZE)
    iter_num = 100


    nodes = [
        [],  # input
        [],  # hidden
        []   # output
    ]

    bias = [
        [],  # ->hidden
        []   # ->output
    ]

    weights = [
        [],  # input -> hidden
        []   # hidden -> output
    ]

    input1, input2 = 0, 0;

    nodes   = [np.zeros(LAYER_SIZE[i])  for i in range(L + 1)]
    weights = [ 2 * np.random.rand(LAYER_SIZE[i + 1], LAYER_SIZE[i]) - 1 for i in range(L)]
    bias    = [ 2 * np.random.rand(LAYER_SIZE[i + 1]) - 1 for i in range(L)]

    total_delta_weights = [np.zeros((LAYER_SIZE[i + 1], LAYER_SIZE[i])) for i in range(L)]
    total_delta_bias    = [np.zeros(LAYER_SIZE[i + 1]) for i in range(L)]


    delta_nodes     = list(nodes)
    delta_bias      = list(bias)
    delta_weights   = list(weights)

    z       = list(nodes)
    delta_z = list(nodes)
    total_delta_weights = list(weights)
    total_delta_bias    = list(bias)

    #endregion

    # region: GET DATA 2
    #orig_data = create_multiple_members(folder_dir)
    orig_data = get_numbers_data()
    inputs = []
    answers = np.zeros((len(orig_data), LAYER_SIZE[L]))
    for i in range(len(orig_data)):
        answers[i][orig_data[i][0]] = 1
        inputs.append(orig_data[i][1])
    # endregion

    # region: GET DATA
    data = Data(folder_dir)
    data = data.members_data
    answers = np.array([])
    inputs  = np.zeros((len(data), np.max([len(data[i]) for i in range(len(data))]) - 1))
    answers  = np.zeros((len(data), LAYER_SIZE[L]))
    endregion


    for i in range(len(inputs)):
        inputs[i] = activation_func(np.array(data[i][1:]))
        ans = [0] * LAYER_SIZE[L]
        ans[data[i][0]] = 1
        answers[i] = ans


    for iter in range(iter_num):
        ef_cost = np.zeros(LAYER_SIZE[L])
        max_cost = np.zeros(LAYER_SIZE[L])
        for j in range(len(inputs)):
            sample = np.array(inputs[j])
            answer = sample
            nodes[0] = activation_func(sample)
            for i in range(1, L + 1):
                z[i] = np.dot(weights[i-1], nodes[i-1]) + bias[i - 1]
                nodes[i] = activation_func(z[i])

            ef_cost += cost(nodes[L], answer) / len(inputs)
            max_cost = np.array([max(m, curr) for m, curr in zip(max_cost, cost(nodes[L], answer))])

            delta_z[L] = cost(nodes[L], answer, True) * activation_func(z[L], True)

            for i in range(1, L + 1):
                delta_bias[L - i]  = np.array(delta_z[L + 1 - i])
                delta_weights[L - i] = np.outer(delta_z[L + 1 - i], nodes[L - i])
                delta_nodes[L - i] = np.dot(delta_z[L + 1 - i], delta_weights[L - i]) / LAYER_SIZE[L - i + 1]
                delta_z[L - i] = delta_nodes[L - i] * activation_func(z[L - i], True)

            total_delta_weights = [np.nan_to_num(tdw + dw) for tdw, dw in zip(total_delta_weights, delta_weights)]
            total_delta_bias    = [np.nan_to_num(tdb + db) for tdb, db in zip(total_delta_bias, delta_bias)]

        weights = [np.nan_to_num(w - DAMP * dw / len(inputs)) for w, dw in zip(weights, total_delta_weights)]
        bias    = [np.nan_to_num(b - DAMP * db / len(inputs)) for b, db in zip(bias, total_delta_bias)]

    return weights, bias, LAYER_SIZE


train_network(folder_dir)


def test_network():
    return