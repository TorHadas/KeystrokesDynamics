import Data
from plotResults import plot
import numpy as np
import os


def activation_func(x, derivative=False):
    if derivative:
        return activation_func(x) * (1.0 - activation_func(x))
    return 1 / (1 + np.exp(-x))


def cost(net_result, answer, derivative=False):
    if derivative:
        return -2 * (answer - net_result)
    return np.abs(np.power(net_result - answer, 2))


class Network():
    # INITIALIZE NEW NETWORK
    def __init__(self, damp=0.0001, default_iter_num=10,
                 network_size=3, layer_size=[15, 11, 11]):
        self.DAMP = damp
        self.DEFAULT_ITER_NUM = default_iter_num
        self.LAYER_SIZE = layer_size
        self.NETWORK_SIZE = network_size
        self.L = network_size - 1
        self.nodes = [np.zeros(self.LAYER_SIZE[i]) for i in range(self.L + 1)]
        self.weights = [2 * np.random.rand(self.LAYER_SIZE[i + 1],
                                           self.LAYER_SIZE[i])
                        - 1 for i
                        in range(self.L)]
        self.bias = [2 * np.random.rand(self.LAYER_SIZE[i + 1]) - 1 for i in
                     range(self.L)]

        self.total_delta_weights = [np.zeros((self.LAYER_SIZE[i + 1],
                                              self.LAYER_SIZE[i])) for i in
                                    range(self.L)]
        self.total_delta_bias = [np.zeros(self.LAYER_SIZE[i + 1]) for i in
                                 range(self.L)]

        self.delta_nodes = list(self.nodes)
        self.delta_bias = list(self.bias)
        self.delta_weights = list(self.weights)

        self.z = list(self.nodes)
        self.delta_z = list(self.nodes)
        self.total_delta_weights = list(self.weights)
        self.total_delta_bias = list(self.bias)
        self.network_train_counter = 0

    def get_data(self, folder_data_dir):
        #orig_data = Data.create_multiple_members(folder_data_dir)
        orig_data = Data.get_fake_data()
        self.answers = np.zeros((len(orig_data), self.LAYER_SIZE[self.L]))
        self.inputs = []
        for i in range(len(orig_data)):
            self.answers[i][orig_data[i][0]] = 1
            self.inputs.append(orig_data[i][1])
        self.data = [self.answers, self.inputs]



    def train_network(self, iter_num=None):
        if(iter_num == None):
            iter_num = self.DEFAULT_ITER_NUM
        self.network_train_counter += 1
        # region: setting variables to be the same as the class variables
        L = self.L; LAYER_SIZE = self.LAYER_SIZE; inputs = self.inputs; answers = self.answers;
        nodes = self.nodes; z = self.z; delta_z = self.delta_z; delta_bias = self.delta_bias; 
        delta_weights = self.delta_weights; delta_nodes = self.delta_nodes; DAMP = self.DAMP; 
        weights = self.weights; total_delta_weights = self.total_delta_weights; bias = self.bias; 
        total_delta_bias = self.total_delta_bias; answers = self.answers;
        #endregion

        # region: setting plot cost vectors
        eff_cost_vec = []
        for i in range(LAYER_SIZE[L]):
            eff_cost_vec.append([0] * iter_num)
        samples_same_ans = [0] * LAYER_SIZE[L]
        # endregion


        for iter in range(iter_num):
            ef_cost = np.zeros(LAYER_SIZE[L])
            max_cost = np.zeros(LAYER_SIZE[L])
            for j in range(len(inputs)):
                sample = np.array(inputs[j])
                answer = np.array(answers[j])

                #CALCULATE NODES
                nodes[0] = activation_func(sample)
                for i in range(1, L + 1):
                    z[i] = np.dot(weights[i - 1], nodes[i - 1]) + bias[i - 1]
                    nodes[i] = activation_func(z[i])

                ef_cost += cost(nodes[L], answer) / len(inputs)
                max_cost = np.array([max(m, curr) for m, curr in zip(max_cost, cost(nodes[L], answer))])

                #LEARN
                delta_z[L] = cost(nodes[L], answer, True) * activation_func(z[L], True)
                for i in range(1, L + 1):
                    delta_bias[L - i] = np.array(delta_z[L + 1 - i])
                    delta_weights[L - i] = np.outer(delta_z[L + 1 - i], nodes[L - i])
                    delta_nodes[L - i] = np.dot(delta_z[L + 1 - i], delta_weights[L - i]) / LAYER_SIZE[L - i + 1]
                    delta_z[L - i] = delta_nodes[L - i] * activation_func(z[L - i], True)

                total_delta_weights = [np.nan_to_num(tdw + dw) for tdw, dw in zip(total_delta_weights, delta_weights)]
                total_delta_bias = [np.nan_to_num(tdb + db) for tdb, db in zip(total_delta_bias, delta_bias)]

                # region: create vector of results to plot
                for i in range(len(answer)):
                    if (answer[i] == 1):
                        # print("in input j: " + str(j) + " the answer is " + str(
                        #    answer) + " the net answer is " + str(nodes[L]) +
                        #      " adding cost to place " + str(i) + " " + str(iter))
                        if (iter == 0):
                            samples_same_ans[i] += 1
                        eff_cost_vec[i][iter] += ((sum(cost(nodes[L], answer)) / LAYER_SIZE[L]) ** (1 / 2))
                # endregion

            weights = [np.nan_to_num(w - DAMP * dw / len(inputs)) for w, dw in zip(weights, total_delta_weights)]
            bias = [np.nan_to_num(b - DAMP * db / len(inputs)) for b, db in zip(bias, total_delta_bias)]

            # normalize by num of samples
            for i in range(len(eff_cost_vec)):
                eff_cost_vec[i][iter] = eff_cost_vec[i][iter] / samples_same_ans[i]
                # endregion

        # region: plot graphs
        iter_vec = []
        for i in range(len(eff_cost_vec)):
            iter_vec.append(np.arange(len(eff_cost_vec[i])))
        plot(iter_vec, eff_cost_vec, "iterations", "normalized cost",
             "costs graphs "
             "damp = " + str(DAMP) + " " + str(self.network_train_counter))
        # endregion

        self.L = L; self.LAYER_SIZE = LAYER_SIZE; self.inputs = inputs; self.answers = answers; self.nodes = nodes; self.z = z; self.delta_z = delta_z; self.delta_bias = delta_bias; self.delta_weights = delta_weights; self.delta_nodes = delta_nodes; self.DAMP = DAMP; self.weights = weights; self.total_delta_weights = total_delta_weights; self.bias = bias; self.total_delta_bias = total_delta_bias; self.answers = answers; 
        return weights, bias, LAYER_SIZE

