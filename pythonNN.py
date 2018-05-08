from sklearn.neural_network import MLPClassifier
import os
import numpy as np
import Data

def get_data(folder_data_dir):
    # orig_data = Data.create_multiple_members(folder_data_dir)
    orig_data = Data.get_fake_data()
    answers = np.zeros((len(orig_data), 5))
    inputs = []
    for i in range(len(orig_data)):
        answers[i][orig_data[i][0]] = 1
        inputs.append(orig_data[i][1])
    data = [answers, inputs]
    print(data)

    return data




folder_dir = os.getcwd() + "\\data"
data = get_data(folder_dir)
clf = MLPClassifier(solver='sgd', alpha=1e-5, random_state = 1, activation='relu')
clf.fit(data[1], data[0])
