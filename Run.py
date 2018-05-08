from NetworkClass import Network
import os
import Data
folder_dir = os.getcwd() + "\\logs"

# network constants
DEFAULT_ITER_NUM = 150
DAMP = 0.05
NETWORK_SIZE = 2
LAYER_SIZE = [
    18,  # input
    2,  # hidden
    8,  # hidden
    2  # output - 11 users, 0 non-user (default)
]

network = Network(DAMP, DEFAULT_ITER_NUM, NETWORK_SIZE, LAYER_SIZE)
network.get_data(folder_dir)
network.train()
network.save("test.json")
net = Network(DAMP, DEFAULT_ITER_NUM, NETWORK_SIZE, LAYER_SIZE)
net.load("test.json")
net.run(Data.vec_to_data([71.7,128.1,120.1,95.8,96.1,-16.1,104.1,200,79.9,624.2,119.8,744.8,95.3,64.5,96.1]))


