import numpy as np
from Data import Data
folder_dir = "bla_bla"

iter_num = 1000
data = Data(folder_dir)
data = data.members_data

def sigmoid (x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def error_calculation(net_result, answer):
    return np.power(net_result-answer, 2)

answer_vector = []
input_data =
for i in range(data.length):
    input_data.append(data[i][1])
    ans = [0]*l0.length
    ans[data[i][0]]=1
    answer_vector.append(ans)
l0 = np.array(l0)
answers = np.array(answer_vector)
l0_length = l0.length
l1_length = 10
l2_length = answer_vector.length

weight0 = 2 * np.random.random((l0_length, l1_length)) - 1
bias0 = np.random.random_sample()

weight1 = 2 * np.random.random((l1_length, l2_length)) - 1
bias1 = np.random.random_sample() + 1

l1 = np.array([0]*l1_length)
l2 = np.array([])
error = 0
for i in range():
    # hidden layers calculation
    l1.add(np.dot(weight0, l0[i]) + bias0)
    l2.add(np.dot(weight1, l1[i]) + bias1)

    # calculating error from answer
    error += error_calculation(l2[i], answer_vector)

