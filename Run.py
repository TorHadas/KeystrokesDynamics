from SimpleNetwork import Network
import os
import Data
import numpy as np
from plotResults import plot
from SimpleNetwork import cost
from SimpleNetwork import activation_func

folder_dir = os.getcwd() + "\\logs"

IMPOSTER = 1
AUTHENTIC = 0
# network constants
DEFAULT_ITER_NUM = 300
DAMP = 0.1
NETWORK_SIZE = 2
LAYER_SIZE = [
    18,  # input
    1,  # hidden
    8,  # hidden
    2  # output - 11 users, 0 non-user (default)
]
#print(activation_func(np.array([78.6,80.9,62.5,99.4,102,-18.4,91.8,357.9,93.2,51,103.4,23.2,112.9,-1.2,85.1])))

'''
network = Network(DAMP, DEFAULT_ITER_NUM, NETWORK_SIZE, LAYER_SIZE)
network.get_data(folder_dir)
network.train()
network.save("test.json")
net = Network(DAMP, DEFAULT_ITER_NUM, NETWORK_SIZE, LAYER_SIZE)
net.load("test.json")
net.run(Data.vec_to_data([71.7,128.1,120.1,95.8,96.1,-16.1,104.1,200,79.9,624.2,119.8,744.8,95.3,64.5,96.1]))
'''

def identification_decider(result):
    maximum = 0
    for i, value in enumerate(result):
        if value > maximum:
            maximum = value
            index = i
    result_answer = np.zeros(len(result))
    result_answer[index] = 1
    return result_answer

def identify(result):
    if(result > 0.8):
        return 1
    return 0

def test_user(user):
    password = user[0]
    print("~ Starting user: " + password + " ~")

    #get data
    all_vectors = Data.to_tuples(user)
    vectors = []
    fake_vectors = []
    for vec in all_vectors:
        if(vec[0] == AUTHENTIC):
            vectors.append(vec)
        else:
            fake_vectors.append(vec)
    train_data = vectors[0:int(len(vectors)/2)] + fake_vectors[0:int(len(fake_vectors)/2)]
    test_data = vectors[int(len(vectors)/2):] + fake_vectors[int(len(fake_vectors)/2):]
    #train network
    network = Network(DAMP, DEFAULT_ITER_NUM, NETWORK_SIZE, LAYER_SIZE)
    network.get_data(train_data)
    network.train(test_data=test_data)
    network.save("save\\"+password+".json")

    count_authentic = 0
    frr = 0 #False Rejection Rate (slightly bad)
    far = 0 #False Acceptance Rate (very very bad)
    mean_cost_vec = []
    max_cost_vec = []

    for test in test_data:
        result, z = network.run(test[1])
        answer = np.zeros(len(result))
        if(test[0] < LAYER_SIZE[1]):
            answer[test[0]] = 1
        #print(result)
        #print(answer)
        mean_cost_vec.append((sum(cost(result, answer)) / len(test)) ** (1/2))
        max_cost_vec.append(max(cost(result, answer)) ** (1/2))
        decision = identification_decider(result)

        if(test[0] == AUTHENTIC):
            if decision[AUTHENTIC]:
                count_authentic += 1
            else:
                frr += 1
        else:
            if decision[AUTHENTIC]:
                far += 1

    iter_vec = np.arange(len(mean_cost_vec))
    x_data = [iter_vec] * 2
    y_data = [mean_cost_vec, max_cost_vec]
    #plot(x_data, y_data, "sample num", "costs", "bla")
    frr = frr / len(test_data)
    far = far / len(test_data)
    print(password + ":\tfrr = " + str(frr) +  "; far = " + str(far))

def test():
    users = Data.get_data()
    for user in users:
        test_user(user)
    

test()
