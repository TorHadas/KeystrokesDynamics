import numpy as np
from Data import Data
folder_dir = "C:\\Users\\t8495554\\Documents\\Workspace\\Code\\KeystrokesDynamics\\data"

#ADD AND SUBSTRACT LISTS
#AL=Infix(lambda x,y: [i + j for i,j in zip(x, y)])
#SL=Infix(lambda x,y: [i - j for i,j in zip(x, y)])
np.set_printoptions(suppress=True,precision=5)
def activation_func(x, derivative=False):
    if derivative:
        return np.nan_to_num(x * (1 - x))
    return 1 / (1 + np.exp(-x))


def cost(net_result, answer, derivative=False):
    if derivative:
        return 2 * (answer - net_result)
    return np.abs(np.power(net_result - answer, 2))

DAMP = 0.03
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


# FOR GRAPHS


# GET DATA
data = Data(folder_dir)
data = data.members_data
answers = np.array([])
inputs  = np.zeros((len(data), np.max([len(data[i]) for i in range(len(data))]) - 1))
answers  = np.zeros((len(data), LAYER_SIZE[L]))


for i in range(len(data)):
    inputs[i] = activation_func(np.array(data[i][1:]))
    ans = [0] * LAYER_SIZE[L]
    ans[data[i][0]] = 1
    answers[i] = ans
last_cost = 0
for iter in range(200):
    
    ef_cost = np.zeros(LAYER_SIZE[L])
    max_cost = np.zeros(LAYER_SIZE[L])
    for j in range(len(inputs)):
        sample = inputs[j]
        answer = answers[j]
        nodes[0] = sample
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
            delta_nodes[L - i] = np.dot(delta_z[L + 1 - i], delta_weights[L - i]) / LAYER_SIZE[L - i + 1]
            delta_z[L - i] = delta_nodes[L - i] * activation_func(z[L - i], True)

        total_delta_weights = [np.nan_to_num(tdw + dw) for tdw, dw in zip(total_delta_weights, delta_weights)]
        total_delta_bias    = [np.nan_to_num(tdb + db) for tdb, db in zip(total_delta_bias, delta_bias)]
    print("EFFECTIVE COST FOR ITER:  " + str(ef_cost))
    print("MAXIMUM OF COST FOR ITER: " + str(max_cost))
    print("SUM EFFECTIVE COST ITER:  " + str(sum(ef_cost**2)))
    print("\t\tCHANGE: \t\t\t" + str(sum(ef_cost**2) - last_cost))
    last_cost = sum(ef_cost**2)
        #print(nodes)
        
    weights = [np.nan_to_num(w - DAMP * dw / len(inputs)) for w, dw in zip(weights, total_delta_weights)]
    bias    = [np.nan_to_num(b - DAMP * db / len(inputs)) for b, db in zip(bias, total_delta_bias)]


# delta_weights[i][j][k] = - DAMP * cost_deriv[j] * activ_deriv[i + 1][j] * nodes[i][j][k]
# delta_weights[i][j] = -DAMP * vec_prod(vec_prod(cost_deriv, activ_deriv[i + 1]), )
# nodes[i][k]




# nodes[i][j] = sigmoid(z[i][j])



