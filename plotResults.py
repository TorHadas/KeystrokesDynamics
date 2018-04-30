import matplotlib.pyplot as plt
import numpy as np
#MPLCONFIGDIR=/tmp/some_pathN python plotResults.py

def plot(x_data, y_data1, y_data2, x_label, y_label, figure_description):
    fig = plt.gcf()
    fig.canvas.set_window_title(figure_description)
    plt.plot(x_data, y_data1, 'r')
    plt.plot(x_data, y_data2, 'g')
    plt.title(y_label + "(" + x_label + ")" )
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
    #plt.savefig(figure_description + ".jpg", nargout=0)


x_data = np.array([1,2,3,4,5,6,7,8,9,10])
y_data1 = 2 * np.array([1,2,3,4,5,6,7,8,9,10])
for i in range(10):
    power = np.random.random_sample(10)
    y_data2 = np.power(np.array([1,2,3,4,5,6,7,8,9,10]), power )

plot(x_data, y_data1, y_data2, 'x', 'y', 'TestToSeeIfTorIsAlert')
