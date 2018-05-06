import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
<<<<<<< HEAD
import os
=======
from functools import reduce

>>>>>>> d14e3051d2d35b3b6fa83a1ed31c89522b8e8887
import numpy as np
#MPLCONFIGDIR=/tmp/some_pathN python plotResults.py

def plot(x_total_data, y_total_data, x_label, y_label, figure_description):
    colors = ["r", "g", "b", "y", "k", "m", "c"]
    fig = plt.gcf()
    fig.canvas.set_window_title(figure_description)
    patches = []
<<<<<<< HEAD
    #print(y_total_data)
    #print(x_total_data)
=======
>>>>>>> d14e3051d2d35b3b6fa83a1ed31c89522b8e8887
    for i in range(len(y_total_data)):
        plt.plot(x_total_data[i], y_total_data[i], colors[i%len(colors)])
        patch = mpatches.Patch(color=colors[i%len(colors)], label='class [' +
                                                             str(i + 1) +
                                                    '] cost')
        patches.append(patch)
    plt.legend(handles=patches)
    plt.title(y_label + "(" + x_label + ")" )
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(os.getcwd() + "\\" + figure_description + \
                ".jpg", bbox_inches='tight')
    #plt.show()

    #plot average cost value
    plt.figure()
    avg_y_data = reduce((lambda x, y: np.array(x) + np.array(y)), y_total_data)
    avg_y_data = np.array(avg_y_data)
    avg_y_data = avg_y_data / len(y_total_data)
    plt.plot(x_total_data[0], avg_y_data, 'r')
    plt.title(y_label + "(" + x_label + ")")
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

    print("saved")

#x_data = np.array([1,2,3,4,5,6,7,8,9,10])
#y_data1 = 2 * np.array([1,2,3,4,5,6,7,8,9,10])
#for i in range(10):
#    power = np.random.random_sample(10)
#    y_data2 = np.power(np.array([1,2,3,4,5,6,7,8,9,10]), power )
#
#plot(x_data, y_data1, y_data2, 'x', 'y', 'TestToSeeIfTorIsAlert')
