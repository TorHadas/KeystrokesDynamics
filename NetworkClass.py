import Data
from Debug import Debug
from plotResults import *
import numpy as np
import os
import random
import json

def activation_func(x, derivative=False):
    if derivative:
        x = activation_func(x) 
        return np.nan_to_num(x * (1.0 - x))
    return 1 / (1 + np.exp(-x))


def cost(net_result, answer, derivative=False):
    if derivative:
        return 2 * (net_result - answer)
    return np.abs(np.power(net_result - answer, 2))


class Network():
    # INITIALIZE NEW NETWORK
    def __init__(self, damp=0.0001, default_iter_num=10,
                 network_size=2, layer_size=[18, 2, 11]):
        np.set_printoptions(precision=4)
        self.debug = Debug()
        #self.debug.off()
        self.DAMP = damp
        self.DEFAULT_ITER_NUM = default_iter_num
        self.LAYER_SIZE = layer_size
        self.NETWORK_SIZE = network_size
        self.L = network_size - 1
        self.nodes = [np.zeros(self.LAYER_SIZE[i]) for i in range(self.L + 1)]
        self.weights = [2 * np.random.rand(self.LAYER_SIZE[i + 1], self.LAYER_SIZE[i]) - 1 for i in range(self.L)]
        self.bias = [2 * np.random.rand(self.LAYER_SIZE[i + 1]) - 1 for i in range(self.L)]

        self.total_delta_weights = [np.zeros((self.LAYER_SIZE[i + 1], self.LAYER_SIZE[i])) for i in range(self.L)]
        self.total_delta_bias = [np.zeros(self.LAYER_SIZE[i + 1]) for i in range(self.L)]

        self.delta_nodes = list(self.nodes)
        self.delta_bias = list(self.bias)
        self.delta_weights = list(self.weights)

        self.z = list(self.nodes)
        self.delta_z = list(self.nodes)
        self.total_delta_weights = list(self.weights)
        self.total_delta_bias = list(self.bias)
        self.network_train_counter = 0

    def save(self, filename):
        data = {
            "weights" : array_to_list(self.weights),
            "bias" : array_to_list(self.bias)
        }
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    def load(self, filename):
        data = {}
        with open(filename, 'r') as file:
            data = json.load(file)
        w = data["weights"]
        b = data["bias"]
        self.weights = [np.array(w[i]) for i in range(len(w))]
        self.bais = [np.array(b[i]) for i in range(len(b))]
    
    def run(self, sample):
        self.nodes[0] = activation_func(sample)
        for i in range(1, self.L + 1):
            self.z[i] = np.dot(self.weights[i - 1], self.nodes[i - 1]) + self.bias[i - 1]
            self.nodes[i] = activation_func(self.z[i])
        return self.nodes[self.L]

    def get_data(self, folder_data_dir):
        orig_data = Data.create_multiple_members(folder_data_dir)
        #orig_data = Data.get_fake_data()
        self.answers = np.zeros((len(orig_data), self.LAYER_SIZE[self.L]))
        self.inputs = []
        for i in range(len(orig_data)):
            self.answers[i][orig_data[i][0]] = 1
            self.inputs.append(orig_data[i][1])
        self.data = [self.answers, self.inputs]



    def train(self, iter_num=None):
        if(iter_num == None):
            iter_num = self.DEFAULT_ITER_NUM
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

        self.debug.log(answers)
        for iter in range(iter_num):
            ef_cost = np.zeros(LAYER_SIZE[L])
            max_cost = np.zeros(LAYER_SIZE[L])
            '''
            combined = list(zip(inputs, answers))
            random.shuffle(combined)
            inputs[:], answers[:] = zip(*combined)
            '''
            self.debug.log(answers)
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

                
                if(j == 1 or j == len(inputs)-1):
                    self.debug.log(nodes[L])
                    self.debug.log(answer)

                total_delta_weights = [np.nan_to_num(tdw + dw) for tdw, dw in zip(total_delta_weights, delta_weights)]
                total_delta_bias    = [np.nan_to_num(tdb + db) for tdb, db in zip(total_delta_bias, delta_bias)]

                # region: create vector of results to plot
                for i in range(len(answer)):
                    if (answer[i] == 1):
                        # self.debug.log("in input j: " + str(j) + " the answer is " + str(
                        #    answer) + " the net answer is " + str(nodes[L]) +
                        #      " adding cost to place " + str(i) + " " + str(iter))
                        if (iter == 0):
                            samples_same_ans[i] += 1
                        eff_cost_vec[i][iter] += ((sum(cost(nodes[L], answer)) / LAYER_SIZE[L]) ** (1 / 2))
                # endregion

            weights = [np.nan_to_num(w - DAMP * dw / len(inputs)) for w, dw in zip(weights, total_delta_weights)]
            bias    = [np.nan_to_num(b - DAMP * db / len(inputs)) for b, db in zip(bias, total_delta_bias)]

            
            #self.debug.log("TOTAL DELTA="+str(DAMP*total_delta_weights[L-1] / len(inputs)))
            
            # normalize by num of samples
            for i in range(len(eff_cost_vec)):
                eff_cost_vec[i][iter] = eff_cost_vec[i][iter] / samples_same_ans[i]
                # endregion

        # region: plot graphs
        iter_vec = []
        for i in range(len(eff_cost_vec)):
            iter_vec.append(np.arange(len(eff_cost_vec[i])))
        simplePlot(iter_vec, eff_cost_vec, DAMP)
        # endregion

        self.L = L; self.LAYER_SIZE = LAYER_SIZE; self.inputs = inputs; self.answers = answers; self.nodes = nodes; self.z = z; self.delta_z = delta_z; self.delta_bias = delta_bias; self.delta_weights = delta_weights; self.delta_nodes = delta_nodes; self.DAMP = DAMP; self.weights = weights; self.total_delta_weights = total_delta_weights; self.bias = bias; self.total_delta_bias = total_delta_bias; self.answers = answers; 
        return weights, bias, LAYER_SIZE


def array_to_list(arr):
    return [np.ndarray.tolist(arr[i]) for i in range(len(arr))],