import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = [1, 7, 12, 19, 27, 35, 45, 53, 67, 72, 82]

# Define the interval bins
bins = [0, 2, 4, 6, 8, 10, 12,14,20,40,60, np.inf]

# Create histogram
hist, bin_edges = np.histogram(data, bins=bins)

# Define the interval labels
interval_labels = ['[0:2]', '[2:4]','[4:6]', '[6:8]', '[8:10]', '[10:12]', '[12:14]', '[14:20]', '[20:40]','[40:60]','[>60]']

# Plot the histogram
plt.bar(interval_labels, hist, edgecolor='black')

# Set labels and title
plt.xlabel('Intervals')
plt.ylabel('Frequency')
plt.title('Histogram with Intervals')

# Display the histogram
plt.show()