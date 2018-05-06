import numpy as np
import Data
from plotResults import *
import os
folder_dir = os.getcwd() + "\\data"

#region: set network constants
DAMP = 0.01
NETWORK_SIZE = 2
L = NETWORK_SIZE - 1
LAYER_SIZE = [
    25,  # input
    5,  # hidden
    5    # output - 8 users, 1 non-user (default)
]
MAX_LAYER_SIZE = np.max(LAYER_SIZE)
ITERATIONS = 200

#endregion

np.set_printoptions(suppress=True,precision=5)
def activation_func(x, derivative=False):
    if derivative:
        return np.nan_to_num(x * (1 - x))
    return 1 / (1 + np.exp(-x))


def cost(net_result, answer, derivative=False):
    if derivative:
        return 2 * (answer - net_result)
    return np.abs(np.power(net_result - answer, 2))


def train_network(data_dir):
    nodes   = [np.zeros(LAYER_SIZE[i])  for i in range(L + 1)]
    weights = [ 2 * np.random.rand(LAYER_SIZE[i + 1], LAYER_SIZE[i]) - 1 for i in range(L)]
    bias    = [ 2 * np.random.rand(LAYER_SIZE[i + 1]) - 1 for i in range(L)]
    z       = list(nodes)

    #total_delta_weights = [np.zeros((LAYER_SIZE[i + 1], LAYER_SIZE[i])) for i in range(L)]
    #total_delta_bias    = [np.zeros(LAYER_SIZE[i + 1]) for i in range(L)]

    delta_nodes     = list(nodes)
    delta_bias      = list(bias)
    delta_weights   = list(weights)    
    delta_z = list(nodes)

    total_delta_weights = list(weights)
    total_delta_bias    = list(bias)

    # region: GET DATA
    data = Data.get_fake_data()

    answers = np.zeros((len(data), LAYER_SIZE[L]))
    inputs = []
    for i in range(len(data)):
        answers[i][data[i][0]] = 1
        inputs.append(data[i][1])
    #endregion

    for i in range(len(data)):
        inputs[i] = activation_func(np.array(data[i][1:]))
        ans = [0] * LAYER_SIZE[L]
        ans[data[i][0]] = 1
        answers[i] = ans
    eff_cost_vec = []
    for i in range(LAYER_SIZE[L]):
        eff_cost_vec.append([0] * ITERATIONS)
    class_samples = [0] * LAYER_SIZE[L]


    for iter in range(ITERATIONS):
        ef_cost = np.zeros(LAYER_SIZE[L])
        max_cost = np.zeros(LAYER_SIZE[L])

        for j in range(len(inputs)):
            sample = inputs[j]
            answer = answers[j]
            nodes[0] = sample[0]

            #UPDATE NODES AND GET OUTPUT
            for i in range(1, L + 1):
                z[i] = np.dot(weights[i - 1], nodes[i - 1]) + bias[i - 1]
                nodes[i] = activation_func(z[i])

            curr_cost = cost(nodes[L], answer);
            ef_cost += curr_cost / len(inputs)
            max_cost = np.array([max(m, curr) for m, curr in zip(max_cost, curr_cost)])
            delta_z[L] = cost(nodes[L], answer, True) * activation_func(z[L], True)

            #DEBUG
            if(j == 5 or j == 200):
                print("iter " + str(iter) + "\t" + str(curr_cost))
 
            #CALCULATE CHANGES IN WEIGHTS AND BIAS
            for i in range(1, L + 1):
                delta_bias[L - i]  = np.array(delta_z[L + 1 - i])
                delta_weights[L - i] = np.outer(delta_z[L + 1 - i], nodes[L - i])
                delta_nodes[L - i] = np.dot(delta_z[L + 1 - i], delta_weights[L - i]) / LAYER_SIZE[L - i + 1]
                delta_z[L - i] = delta_nodes[L - i] * activation_func(z[L - i], True)

            total_delta_weights = [np.nan_to_num(tdw + dw) for tdw, dw in zip(total_delta_weights, delta_weights)]
            total_delta_bias    = [np.nan_to_num(tdb + db) for tdb, db in zip(total_delta_bias, delta_bias)]

            #region: create vector of results to plot
            for i in range(LAYER_SIZE[L]):
                if(answer[i] == 1):
                    if(iter == 0):
                        class_samples[i] += 1
                    eff_cost_vec[i][iter] += (sum(curr_cost) / LAYER_SIZE[L]) ** (1/2)
            #endregion

        weights = [np.nan_to_num(w - DAMP * dw / len(inputs)) for w, dw in zip(weights, total_delta_weights)]
        bias    = [np.nan_to_num(b - DAMP * db / len(inputs)) for b, db in zip(bias, total_delta_bias)]

        # normalize by num of samples
        for i in range(len(eff_cost_vec)):
            eff_cost_vec[i][iter] = eff_cost_vec[i][iter] / class_samples[i]
        #endregion

    #region: plot graphs
    iter_vec = []
    for i in range(len(eff_cost_vec)):
        iter_vec.append(np.arange(len(eff_cost_vec[i])))
    simplePlot(iter_vec, eff_cost_vec, DAMP)
    #endregion
    return weights, bias, LAYER_SIZE


train_network(folder_dir)