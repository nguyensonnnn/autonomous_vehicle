import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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

x3=[]
x4=[]

#with open("model_600_steering.txt") as f2:
 #   for line in f2:
  #      line_values = line.split()
   #     if len(line_values) >= 5:
    #        x2.append(float(line_values[4]))
#with open("driving_dataset/data.txt") as f1:
 #   i=0
  #  for line in f1:
   #     if i%2==0:
    #        x3.append(float(line.split()[5]))
     #   if i%2==1:
      #      x4.append(float((line.split()[5])))
    #i+=1
t=list(range(len(x3))) 
i=0       
with open("training.txt") as f3: 
    for line in f3:
        print(i)
        if i%2==0:
            line_values = line.split()
            if len(line_values) >= 5:
                x3.append(float(line.split()[4])*1.25)
        if i%2==1:
            line_values = line.split()
            if len(line_values) >= 4:
                x4.append(float(line.split()[3]))
        i+=1
print(len(x3))    
print(len(x4))    
        
x = list(range(len(x3)))  # Assuming x1 and x2 have the same length

#x_label = "X-axis"
#y_label = "Y-axis"
#title = "Two Lines Graph"
#legend_labels = ["Line 1", "Line 2"]

#draw_line_graph([x, x], [x1, x2], x_label, y_label, title, legend_labels)

x_label= "epoch"
y_label="loss"
title="Training loss and Validation loss throughout the training period"
legend_labels= ["Training Loss","Validation Loss"]
draw_line_graph([x, x], [x3, x4], x_label, y_label, title, legend_labels)