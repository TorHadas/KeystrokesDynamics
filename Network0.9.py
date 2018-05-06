import numpy as np
from Data import Data
folder_dir = "C:\\Users\\t8495554\\Documents\\Workspace\\Code\\KeystrokesDynamics\\data"
# make it for vectors

def activation_func(x, derivative=False):
    if derivative:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def cost(net_result, answer, derivative=False):
    if derivative:
        return 2*(net_result - answer)
    return np.abs(np.power(net_result - answer, 2))

DAMP = 0.01
NETWORK_SIZE = 2
L = NETWORK_SIZE - 1
LAYER_SIZE = [
    25,  # input
    5,  # hidden
    5    # output - 8 users, 1 non-user (default)
]
MAX_LAYER_SIZE = np.max(LAYER_SIZE)


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
nodes   = [nodes.append(np.zeros(LAYER_SIZE[i]))  for i in range(L + 1)]
bias    = [bias.append(np.zeros(LAYER_SIZE[i]))       for i in range(L)]
weights = [np.zeros((LAYER_SIZE[i + 1], LAYER_SIZE[i])) for i in range(L)]
delta_nodes     = list(nodes)
delta_bias      = list(bias)
delta_weights   = list(weights)

z       = list(nodes)
delta_z = list(nodes)
total_delta_weights = list(weights)
total_delta_bias    = list(bias)



# GET DATA
data = Data(folder_dir)
data = data.members_data
answers = np.array([])
inputs  = np.zeros((len(data), np.max([len(data[i]) for i in range(len(data))]) - 1))
answers  = np.zeros((len(data), LAYER_SIZE[L]))


for i in range(len(data)):
    inputs[i] = data[i][1:]
    ans = [0] * LAYER_SIZE[L]
    ans[data[i][0]] = 1
    answers[i] = ans


weights = [ 2 * np.random.rand(LAYER_SIZE[i + 1], LAYER_SIZE[i]) - 1 for i in range(L)]
bias    = [ 2 * np.random.rand(LAYER_SIZE[i + 1]) - 1 for i in range(L)]

total_delta_weights = [np.zeros((LAYER_SIZE[i + 1], LAYER_SIZE[i])) for i in range(L)]
total_delta_bias    = [np.zeros(LAYER_SIZE[i + 1]) for i in range(L)]

for iter in range(30):
    for j in range(len(inputs)):
        sample = inputs[j]
        answer = answers[j]
        nodes[0] = sample
        activ_deriv = np.array([])
        for i in range(1, L + 1):    
            z[i] = np.dot(weights[i-1], nodes[i-1]) + bias[i - 1]
            nodes[i] = activation_func(z[i])

        delta_z[L] = cost(nodes[L], answer, True) * activation_func(z[L], True)
        '''
            delta_bias[L - 1] = np.array(delta_z[L])
            delta_weights[L - 1] = np.outer(delta_z[L], nodes[L - 1])
            
            delta_nodes[L - 1] = np.dot(delta_weights[L - 1], delta_z[L]) / LAYER_SIZE[L]
            delta_z[L - 1] = delta_nodes[L - 1] * activation_func(z[L - 1], True)
            delta_bias[L - 2]  = np.array(delta_z[L - 1])
            delta_weights[L - 2] = np.outer(delta_z[L - 1], nodes[L - 2])
        '''
        for i in range(1, L + 1):
            delta_bias[L - i]  = np.array(delta_z[L + 1 - i])
            delta_weights[L - i] = np.outer(delta_z[L + 1 - i], nodes[L - i])
            delta_nodes[L - i] = np.dot(delta_weights[L - i], delta_z[L + 1 - i]) / LAYER_SIZE[L - i + 1]
            delta_z[L - i] = delta_nodes[L - 1] * activation_func(z[L - i], True)

        total_delta_weights += delta_weights
        total_delta_bias    += delta_bias

        #print(nodes)
        print(str(iter) + "::" + str(j) + str(nodes[L]))
    weights -= DAMP * total_delta_weights / len(inputs)
    bias    -= DAMP *  total_delta_bias / len(inputs)


# delta_weights[i][j][k] = - DAMP * cost_deriv[j] * activ_deriv[i + 1][j] * nodes[i][j][k]
# delta_weights[i][j] = -DAMP * vec_prod(vec_prod(cost_deriv, activ_deriv[i + 1]), )
# nodes[i][k]




# nodes[i][j] = sigmoid(z[i][j])



