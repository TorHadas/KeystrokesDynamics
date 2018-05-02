import numpy as np
from Data import *
from plotResults import plot
#folder_dir = "C:\\Users\\T8497069\\Desktop\\Smop\\KeystrokesDynamics\\data"
folder_dir = 'C:\\Users\\T8497069\\Desktop\\Smop\\KeystrokesDynamics\\logs'


np.set_printoptions(suppress=True,precision=5)
def activation_func(x, derivative=False):
    if derivative:
        return np.nan_to_num(x * (1 - x))
    return 1 / (1 + np.exp(-x))


def cost(net_result, answer, derivative=False):
    if derivative:
        return 2 * (answer - net_result)
    return np.abs(np.power(net_result - answer, 2))


def train_network(folder_data_dir):

    #region: set network constants
    DAMP = 0.001
    NETWORK_SIZE = 3
    L = NETWORK_SIZE - 1
    LAYER_SIZE = [
        15,  # input
        13,  # hidden
        11    # output - 11 users, 0 non-user (default)
    ]
    MAX_LAYER_SIZE = np.max(LAYER_SIZE)
    iter_num = 150


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
    orig_data = create_multiple_members(folder_dir)
    inputs = []
    answers = np.zeros((len(orig_data), LAYER_SIZE[L]))
    print(len(answers), len(answers[0]))
    for i in range(len(orig_data)):
        print(i, orig_data[i][0])
        answers[i][orig_data[i][0]] = 1
        inputs.append(orig_data[i][1])
    # endregion

    # region: GET DATA
    #data = Data(folder_dir)
    #data = data.members_data
    #answers = np.array([])
    #inputs  = np.zeros((len(data), np.max([len(data[i]) for i in range(len(
    # data))]) - 1))
    #answers  = np.zeros((len(data), LAYER_SIZE[L]))
    #endregion


    #for i in range(len(inputs)):
    #    inputs[i] = activation_func(np.array(data[i][1:]))
    #    ans = [0] * LAYER_SIZE[L]
    #    ans[data[i][0]] = 1
    #    answers[i] = ans
    eff_cost_vec = []
    for i in range(LAYER_SIZE[L]):
        eff_cost_vec.append([0] * iter_num)
    samples_same_ans = [0] * LAYER_SIZE[L]


    for iter in range(iter_num):
        ef_cost = np.zeros(LAYER_SIZE[L])
        max_cost = np.zeros(LAYER_SIZE[L])
        for j in range(len(inputs)):
            print(inputs[j])
            sample = np.array(inputs[j])
            answer = np.array(answers[j])
            nodes[0] = activation_func(sample)
            for i in range(1, L + 1):
                z[i] = np.dot(weights[i-1], nodes[i-1]) + bias[i - 1]
                nodes[i] = activation_func(z[i])
            ef_cost += cost(nodes[L], answer) / len(inputs)
            max_cost = np.array([max(m, curr) for m, curr in zip(max_cost, cost(nodes[L], answer))])
            #if j == 1 or j == 400:# or j == 800:
                #print(str(iter) + "::" + str(j) + "\t" + str(nodes[L]))
                #print(str(iter) + "\t" + str(j) + str(weights))
                #print(str(iter) + "::" + str(j) + "\t" + str(cost(nodes[L], answer)))
                #print("~~~~~\t~~~~~~~~~~")

            delta_z[L] = cost(nodes[L], answer, True) * activation_func(z[L], True)

            #region: calculations
            '''
                delta_bias[L - 1] = np.array(delta_z[L])
                delta_weights[L - 1] = np.outer(delta_z[L], nodes[L - 1])
                
                delta_nodes[L - 1] = np.dot(delta_weights[L - 1], delta_z[L]) / LAYER_SIZE[L]
                delta_z[L - 1] = delta_nodes[L - 1] * activation_func(z[L - 1], True)
                delta_bias[L - 2]  = np.array(delta_z[L - 1])
                delta_weights[L - 2] = np.outer(delta_z[L - 1], nodes[L - 2])
            '''
            #endregion

            for i in range(1, L + 1):
                delta_bias[L - i]  = np.array(delta_z[L + 1 - i])
                delta_weights[L - i] = np.outer(delta_z[L + 1 - i], nodes[L - i])
                delta_nodes[L - i] = np.dot(delta_z[L + 1 - i], delta_weights[L - i]) / LAYER_SIZE[L - i + 1]
                delta_z[L - i] = delta_nodes[L - i] * activation_func(z[L - i], True)

            total_delta_weights = [np.nan_to_num(tdw + dw) for tdw, dw in zip(total_delta_weights, delta_weights)]
            total_delta_bias    = [np.nan_to_num(tdb + db) for tdb, db in zip(total_delta_bias, delta_bias)]

            #region: create vector of results to plot
            for i in range(len(answer)):
                if(answer[i] == 1):
                   #print("in input j: " + str(j) + " the answer is " + str(
                   #    answer) + " the net answer is " + str(nodes[L]) +
                    #      " adding cost to place " + str(i) + " " + str(iter))
                    if(iter == 0):
                        samples_same_ans[i] += 1
                    eff_cost_vec[i][iter] += ((sum(cost(nodes[L], answer)) /
                                               LAYER_SIZE[L])  ** (1/2))
            #endregion

        weights = [np.nan_to_num(w - DAMP * dw / len(inputs)) for w, dw in zip(weights, total_delta_weights)]
        bias    = [np.nan_to_num(b - DAMP * db / len(inputs)) for b, db in zip(bias, total_delta_bias)]

        # normalize by num of samples
        for i in range(len(eff_cost_vec)):
            eff_cost_vec[i][iter] = eff_cost_vec[i][iter] / samples_same_ans[i]
        #endregion

    #region: plot graphs
    iter_vec = []
    for i in range(len(eff_cost_vec)):
        iter_vec.append(np.arange(len(eff_cost_vec[i])))
    plot(iter_vec, eff_cost_vec, "iterations", "normalized cost", "costs graphs "
                                                        "damp = " + str(DAMP) )
    #endregion
    return weights, bias, LAYER_SIZE


train_network(folder_dir)


def test_network():
    return