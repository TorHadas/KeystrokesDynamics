from NetworkClass import Network
import os

folder_dir = os.getcwd() + "\\logs"

# network constants
DEFAULT_ITER_NUM = 200
DAMP = 0.001
NETWORK_SIZE = 3
LAYER_SIZE = [
    25,  # input
    11,  # hidden
    5  # output - 11 users, 0 non-user (default)
]
ITER_NUM = 200

network = Network(DAMP, ITER_NUM, NETWORK_SIZE, LAYER_SIZE)
network.get_data(folder_dir)
network.train_network()
