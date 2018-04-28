import numpy as np
from Data import Data
folder_dir = "bla_bla"

damp = 0.5

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

delta_nodes = [[], [], []]
delta_bias = [[], []]
delta_weights = [[], []]

# delta_weights[i][j][k] = - damp * cost_deriv[j] * activ_deriv[i][j] *
# nodes[i][k]


# z[i][j] = SUM_k(weights[i-1][j][k]*nodes[i-1][k]) + bias[i][j];
# z[i][j] = weights[i-1][j]*nodes[i-1] + bias[i][j];
# nodes[i][j] = sigmoid(z[i][j]);


def activation_func(x, derivative=False):
    if derivative:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def cost(net_result, answer, derivative=False):
    if derivative:
        return 2*(net_result - answer)
    return np.power(net_result - answer, 2)






iter_num = 1000
data = Data(folder_dir)
data = data.members_data
answer_vector = []
input_data = []

for i in range(data.length):
    input_data.append(data[i][1])
    ans = [0]*input_data[i].length
    ans[data[i][0]] = 1
    answer_vector.append(ans)

l0 = np.array(input_data[0])
answers = np.array(answer_vector)
l0_length = len(l0)
l1_length = 10
l2_length = len(answer_vector)

weight0 = 2 * np.random.random((l0_length, l1_length)) - 1
bias0 = np.random.random_sample()

weight1 = 2 * np.random.random((l1_length, l2_length)) - 1
bias1 = np.random.random_sample() + 1

l1 = np.array([])
l2 = np.array([])
error = 0
for sample in input_data:
    l0 = sample
    # hidden layers calculation
    l1 = (np.dot(weight0, l0) + bias0)
    l2 = (np.dot(weight1, l1) + bias1)

    # calculating error from answer
    error += cost(l2, answer_vector[i])




