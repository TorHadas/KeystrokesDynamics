from NetworkClass import Network
import os

folder_dir = os.getcwd() + "\\logs"

# network constants
DEFAULT_ITER_NUM = 30
DAMP = 0.1
NETWORK_SIZE = 2
LAYER_SIZE = [
    25,  # input
    5,  # hidden
    5  # output - 11 users, 0 non-user (default)
]

network = Network(DAMP, DEFAULT_ITER_NUM, NETWORK_SIZE, LAYER_SIZE)
network.get_data(folder_dir)
network.train_network()
