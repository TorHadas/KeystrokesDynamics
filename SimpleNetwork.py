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
    POWER = 2;
    if derivative:
        return POWER * np.power(net_result - answer, POWER - 1)
    return np.abs(np.power(net_result - answer, POWER))

def identify(result):
    if(result > 0.8):
        return 1
    return 0

class Network():
    # INITIALIZE NEW NETWORK
    def __init__(self, damp=0.0001, default_iter_num=10,
                 network_size=2, layer_size=[18, 1]):
        np.set_printoptions(precision=4)
        self.debug = Debug()
        self.debug.off()
        self.DAMP = damp
        self.DEFAULT_ITER_NUM = default_iter_num
        self.LAYER_SIZE = layer_size
        self.NETWORK_SIZE = network_size
        self.L = network_size - 1
        self.OUTPUT_SIZE = 1
        self.nodes = [np.zeros(self.LAYER_SIZE[i]) for i in range(self.L + 1)]
        self.weights = [0.01 * (np.random.rand(self.LAYER_SIZE[i + 1], self.LAYER_SIZE[i]) - 0.5) for i in range(self.L)]
        self.bias    = [0.01 * (np.random.rand(self.LAYER_SIZE[i + 1]) - 0.5) for i in range(self.L)]

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
        if(len(sample) != len(self.nodes[0])):
            print("ERROR - sample length does not match first layer size.\nSample: "+str(sample))
        self.nodes[0] = activation_func(sample)
        #self.nodes[0] = sample
        z = list(self.nodes)
        for i in range(1, self.L + 1):
            z[i] = np.dot(self.weights[i - 1], self.nodes[i - 1]) + self.bias[i - 1]
            self.nodes[i] = activation_func(z[i])
        return self.nodes[self.L], z

    def get_data(self, data):
        self.answers = np.zeros(len(data))
        self.inputs = []
        for i in range(len(data)):
            self.answers[i] = 1 - data[i][0]
            self.inputs.append(data[i][1])
        self.data = [self.answers, self.inputs]

    def cal_samples(self, answers):
        samples_same_ans = [0, 0]
        for j in range(len(answers)):
            answer = np.array(answers[j])
            samples_same_ans[1 - int(answer)] += 1
        return samples_same_ans

    def train(self, iter_num=None, test_data=None):
        testing = not test_data is None

        if testing:
            test_inputs = [test_data[j][1] for j in range(len(test_data))]
            test_answers = [1 - test_data[j][0]  for j in range(len(test_data))]

        if(iter_num == None):
            iter_num = self.DEFAULT_ITER_NUM
        # region: setting variables to be the same as the class variables

        total_delta_weights = [np.zeros((self.LAYER_SIZE[i + 1], self.LAYER_SIZE[i])) for i in range(self.L)]
        total_delta_bias = [np.zeros(self.LAYER_SIZE[i + 1]) for i in range(self.L)]

        delta_nodes = list(self.nodes)
        delta_bias = list(self.bias)
        delta_weights = list(self.weights)

        z = list(self.nodes)
        delta_z = list(self.nodes)
        total_delta_weights = list(self.weights)
        total_delta_bias = list(self.bias)
        #endregion

        # region: setting plot cost vectors
        eff_cost_vec = [[0] * iter_num for i in range(self.LAYER_SIZE[self.L] + 1)]
        test_cost_vec = list(eff_cost_vec)
        wrong_vec = [0] * iter_num

        samples_same_ans = self.cal_samples(self.answers)
        tests_same_ans = self.cal_samples(test_answers)

        
        # endregion

        for iter in range(iter_num):
            combined = list(zip(self.inputs, self.answers))
            random.shuffle(combined)
            inputs, answers = zip(*combined)

            for j in range(len(inputs)):
                sample = np.array(inputs[j])
                answer = answers[j]

                #CALCULATE NODES
                a, z = self.run(sample)

                #LEARN

                delta_z[self.L] = cost(self.nodes[self.L], answer, True) * activation_func(z[self.L], True)
                for i in range(1, self.L + 1):
                    delta_bias[self.L - i] = np.array(delta_z[self.L + 1 - i])
                    delta_weights[self.L - i] = np.outer(delta_z[self.L + 1 - i], self.nodes[self.L - i])
                    delta_nodes[self.L - i] = np.dot(delta_z[self.L + 1 - i], delta_weights[self.L - i]) / self.LAYER_SIZE[self.L - i + 1]
                    delta_z[self.L - i] = delta_nodes[self.L - i] * activation_func(z[self.L - i], True)

                
                if(j == 1 or j == len(inputs)-1):
                    self.debug.log("j="+str(j)+", "+str(self.nodes[self.L]))
                    self.debug.log("answer = "+str(answer))

                total_delta_weights = [np.nan_to_num(tdw + dw) for tdw, dw in zip(total_delta_weights, delta_weights)]
                total_delta_bias    = [np.nan_to_num(tdb + db) for tdb, db in zip(total_delta_bias, delta_bias)]

                eff_cost_vec[1-int(answer)][iter] += sum(np.abs(self.nodes[self.L] - answer))
                if(identify(self.nodes[self.L]) == answer):
                    wrong_vec[iter] += 1

            for j in range(len(test_inputs)):
                sample = np.array(test_inputs[j])
                answer = test_answers[j]
                a, z = self.run(sample)
                test_cost_vec[1-int(answer)][iter] += sum(np.abs(self.nodes[self.L] - answer))
                if(identify(self.nodes[self.L]) == answer):
                    wrong_vec[iter] += 1
                # endregion

            self.weights = [np.nan_to_num(w - self.DAMP * dw / len(inputs)) for w, dw in zip(self.weights, total_delta_weights)]
            self.bias    = [np.nan_to_num(b - self.DAMP * db / len(inputs)) for b, db in zip(self.bias, total_delta_bias)]

            
            #self.debug.log("TOTAL DELTA="+str(DAMP*total_delta_weights[self.L-1] / len(inputs)))
            
            # normalize by num of samples
            for i in range(len(eff_cost_vec)):
                eff_cost_vec[i][iter] = eff_cost_vec[i][iter] / samples_same_ans[i]
                # endregion
        # region: plot graphs
        iter_vec = []
        for i in range(len(eff_cost_vec)):
            iter_vec.append(np.arange(len(eff_cost_vec[i])))

        if testing:
            y = np.array(eff_cost_vec[0])
            y2 = np.array(eff_cost_vec[1])
            y3 = np.array(wrong_vec)
            plot3(iter_vec[0], y, y2, y3, "Iterations", "Cost", str(self.DAMP))
            y = np.array(test_cost_vec[0])
            y2 = np.array(test_cost_vec[1])
            
            plot3(iter_vec[0], y, y2, y3, "Iterations", "Cost", str(self.DAMP))
        # endregion

        return self.weights, self.bias, self.LAYER_SIZE


def array_to_list(arr):
    return [np.ndarray.tolist(arr[i]) for i in range(len(arr))],