import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def draw_line_graph(x_values_list, y_values_list, x_label, y_label, title, legend_labels):
    """
    Draws a line graph given multiple sets of x and y values.

    Args:
        x_values_list (list of lists or arrays): List of x-axis values for each line.
        y_values_list (list of lists or arrays): List of y-axis values for each line.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the graph.
        legend_labels (list of str): Labels for the legend indicating the lines.
    """
    for i in range(len(x_values_list)):
        plt.plot(x_values_list[i], y_values_list[i], marker='o', label=legend_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    x_min = min(min(x_values) for x_values in x_values_list)
    x_max = max(max(x_values) for x_values in x_values_list)
    y_min = min(min(y_values) for y_values in y_values_list)
    y_max = 100
    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    plt.show()
x1 = []
x2 = []
def average(list_function):
    sum_value=0
    for element in list_function:
        sum_value+=element
    average=sum_value/len(list_function)
    return average
with open("plot.txt") as f2:
    for line in f2:
        line_values = line.split()
        if len(line_values) >= 5:
            x2.append(float(line_values[4]))
with open("testing_set/data.txt") as f1:
    i=0
    for line in f1:
        if i>=44100:
            x1.append(float(line.split()[1][0:8]))
        i+=1
        if i==46636:
            break
        
dx=[]
for i in range(len(x1)):
    di=abs(x1[i]-x2[i])
    dx.append(di)
print(average(dx))
x = list(range(len(x1)))  # Assuming x1 and x2 have the same length

x_label = "frame"
y_label = "steering angle"
title = "Comparison of predict steering command and acutal steering command"
legend_labels = ["Actual steering command", "predict steering command"]

draw_line_graph([x, x], [x1, x2], x_label, y_label, title, legend_labels)

x1_label = "frame"
y1_label = "steering angle error"
title1 = "Error of prediction"
legend_labels1 = ["Steering command error"]

draw_line_graph([x, x], [x1, x2], x_label, y_label, title, legend_labels)
draw_line_graph([x], [dx], x1_label, y1_label, title1, legend_labels1)



# Sample data


# Define the interval bins
bins = [0, 5, 10, 15, 20, 25,30,60, np.inf]

# Create histogram
hist, bin_edges = np.histogram(dx, bins=bins)

# Define the interval labels
interval_labels = ['[0:5]', '[5:10]','[10:15]', '[15:20]', '[20:25]', '[25:30]', '[30:60]', '[>60]']

# Plot the histogram
plt.bar(interval_labels, hist, edgecolor='black')

# Set labels and title
plt.xlabel('error')
plt.ylabel('Frequency')
plt.title('Histogram of prediction error')

# Display the histogram
plt.show()
