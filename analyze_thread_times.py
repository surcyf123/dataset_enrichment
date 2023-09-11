import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV into a dataframe
df = pd.read_csv('thread_times.csv', header=None, names=['Model', 'Time', 'Characters'])

# Separate out the "Total" rows for overall analysis
total_rows = df[df['Model'] == 'Total']
df = df[df['Model'] != 'Total']

# Calculate mean, min, and max time
mean_time = df['Time'].mean()
min_time = df['Time'].min()
max_time = df['Time'].max()

print(f"Mean Time: {mean_time} seconds")
print(f"Min Time: {min_time} seconds")
print(f"Max Time: {max_time} seconds")

# Calculate time per character
df['TimePerChar'] = df['Time'] / df['Characters']

# Plot
plt.scatter(df['Characters'], df['TimePerChar'])
plt.title('Extra Time Needed per Character')
plt.xlabel('Number of Characters')
plt.ylabel('Time per Character (seconds)')
plt.grid(True)
plt.show()