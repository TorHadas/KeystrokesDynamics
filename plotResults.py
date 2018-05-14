import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functools import reduce
import datetime as dt
import os
import numpy as np
#MPLCONFIGDIR=/tmp/some_pathN python plotResults.py
def plot3(x_total_data, y_total_data, y2, y3, x_label, y_label, figure_description):
    time = str(dt.datetime.now().day) + "  " + str(dt.datetime.now().hour) + " " + str(dt.datetime.now().minute)
    colors = ["r", "g", "b", "y", "k", "m", "c"]
    fig = plt.gcf()
    fig.canvas.set_window_title(figure_description)
    patches = []
    plt.plot(x_total_data[0], y_total_data[0], "r")
    plt.plot(x_total_data[0], y2, "g")
    plt.plot(x_total_data[0], y3, "b")
    patch = mpatches.Patch(color="r", label='train cost')
    patch = mpatches.Patch(color="g", label='test cost')
    patch = mpatches.Patch(color="b", label='correct')
    patches.append(patch)
    plt.legend(handles=patches)
    plt.title(y_label + "(" + x_label + ")" + " " + time)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(os.getcwd() + "\\plots\\" + figure_description + \
                ".jpg", bbox_inches='tight')
    #plt.show()

    #plot average cost value
    '''
    plt.figure()
    avg_y_data = reduce((lambda x, y: np.array(x) + np.array(y)), y_total_data)
    avg_y_data = np.array(avg_y_data)
    avg_y_data = avg_y_data / len(y_total_data)
    plt.plot(x_total_data[0], avg_y_data, 'r')
    plt.title("mean " + y_label + "(" + x_label + ")" + " " + time)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.savefig(os.getcwd() + "\\plots\\mean" + figure_description + \
        ".jpg", bbox_inches='tight')
    '''
    plt.show()

def plot(x_total_data, y_total_data, x_label, y_label, figure_description, y2=None, y3=None):
    time = str(dt.datetime.now().day) + "  " + str(dt.datetime.now().hour) + " " + str(dt.datetime.now().minute)
    colors = ["r", "g", "b", "y", "k", "m", "c"]
    fig = plt.gcf()
    fig.canvas.set_window_title(figure_description)
    patches = []
    for i in range(len(y_total_data)):
        plt.plot(x_total_data[i], y_total_data[i], colors[i%len(colors)])
        patch = mpatches.Patch(color=colors[i%len(colors)], label='class [' +
                                                             str(i + 1) +
                                                    '] cost')
        patches.append(patch)
    plt.legend(handles=patches)
    plt.title(y_label + "(" + x_label + ")" + " " + time)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(os.getcwd() + "\\plots\\" + figure_description + \
                ".jpg", bbox_inches='tight')
    #plt.show()

    #plot average cost value
    '''
    plt.figure()
    avg_y_data = reduce((lambda x, y: np.array(x) + np.array(y)), y_total_data)
    avg_y_data = np.array(avg_y_data)
    avg_y_data = avg_y_data / len(y_total_data)
    plt.plot(x_total_data[0], avg_y_data, 'r')
    plt.title("mean " + y_label + "(" + x_label + ")" + " " + time)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.savefig(os.getcwd() + "\\plots\\mean" + figure_description + \
        ".jpg", bbox_inches='tight')
    '''
    plt.show()




def simplePlot (iterations_vector, effective_cost_vector, DAMP):
    time = str(dt.datetime.now().day) + "  " + str(dt.datetime.now().hour) + " " + str(dt.datetime.now().minute)
    plot(iterations_vector, effective_cost_vector, "iterations", "normalized cost", "costs graphs damp "
         + str(DAMP) + " " + time)
