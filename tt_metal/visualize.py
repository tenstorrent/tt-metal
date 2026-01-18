# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Set parameters
# num_timesteps = 5
# grid_size = 10

# # Create a random dataset for the heatmap (replace with your own data)
# data = np.random.rand(num_timesteps, grid_size, grid_size)

# # Initialize the figure and axis
# fig, ax = plt.subplots()

# # Create an initial heatmap
# heatmap = ax.imshow(data[0], cmap='viridis', vmin=0, vmax=1)

# # Function to update the heatmap
# def update(frame):
#     heatmap.set_array(data[frame])
#     ax.set_title(f'Timestep {frame}')
#     return heatmap,

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_timesteps, blit=True)

# # Show the animation
# plt.colorbar(heatmap)
# plt.show()

device_locs = {
    #  x,y
   4: [3,6],
   5: [3,5],
   6: [2,5],
   7: [2,6],
   8: [1,6],
   9: [1,7],
   10: [2,7],
   11: [3,7],
   12: [0,7],
   13: [0,6],
   14: [0,5],
   15: [1,5],
   16: [1,4],
   17: [2,4],
   18: [3,4],
   19: [3,3],
   20: [2,3],
   21: [1,3],
   22: [1,2],
   23: [2,2],
   24: [3,2],
   25: [3,1],
   26: [2,1],
   27: [1,1],
   28: [1,0],
   29: [2,0],
   30: [3,0],
   31: [0,0],
   32: [0,1],
   33: [0,2],
   34: [0,3],
   35: [0,4],
}


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load data from a CSV file
csv_file = '/Users/snijjar/Desktop/TG_all_gather_cols.csv'  # Replace with your CSV file path
# csv_file = '/Users/snijjar/Desktop/TG_all_gather_rows.csv'  # Replace with your CSV file path
dataframe = pd.read_csv(csv_file)

# Define your filtering criteria
# Example: Get rows where 'ColumnA' == 'some_value' and select specific columns
filter_column = 'OP CODE'
filter_value = 'AllGather'
measured_column = 'OP TO OP LATENCY [ns]'
# selected_columns = ['GLOBAL CALL COUNT', 'DEVICE ID', 'DEVICE FW DURATION [ns]']  # Replace with your actual column names
selected_columns = ['GLOBAL CALL COUNT', 'DEVICE ID', measured_column]  # Replace with your actual column names

# Filter the dataframe
op_data = dataframe[dataframe[filter_column] == filter_value][selected_columns]
print(op_data)

galaxy_devices = list(range(4,36))
device_times = {}
for device_id in galaxy_devices:
    # remaining_columns = ['GLOBAL CALL COUNT', 'DEVICE FW DURATION [ns]']  # Replace with your actual column names
    remaining_columns = ['GLOBAL CALL COUNT', measured_column]  # Replace with your actual column names
    device_data = dataframe[(dataframe[filter_column] == filter_value) & (dataframe['DEVICE ID'] == device_id)][remaining_columns]
    print(len(device_data))
    print(device_data)
        
    sorted_data = device_data.sort_values(by='GLOBAL CALL COUNT')
    # last_column_values = sorted_data['DEVICE FW DURATION [ns]'].values[3:]
    last_column_values = sorted_data[measured_column].values[4:32]
    device_times[device_id] = last_column_values
    
    print (f"values: {last_column_values}")
        
grid_x = 4
grid_y = 8

# Convert the filtered data to a 3D NumPy array for heatmap generation
# Assuming you want to reshape it based on the number of timesteps and grid size
num_timesteps = len(list(device_times.values())[0])
grid_size = grid_x * grid_y

# Create an empty array and fill it
data = []
for t in range(num_timesteps):
    data.append(np.zeros((grid_y, grid_x)))
# data = np.zeros((num_timesteps, grid_y, grid_x))

smallest = 1000000000000
largest = 0
for core, vals in device_times.items():
    x, y = device_locs[core]
    for i, val in enumerate(vals):
        data[i][y, x] = val
        largest = val if val > largest else largest
        smallest = val if val < smallest else smallest

# for i, row in enumerate(filtered_data.itertuples(index=False)):
#     # Example: Populate the data array based on specific columns
#     data[i] = np.array(row).reshape((grid_size, grid_size))  # Adjust this as needed


# Initialize the figure and axis for heatmap
fig, ax = plt.subplots()
heatmap = ax.imshow(data[0], cmap='viridis', vmin=smallest, vmax=largest)

# Update function for animation
def update(frame):
    heatmap.set_array(data[frame])
    ax.set_title(f'Timestep {frame}')
    return heatmap,

for t in range(num_timesteps):
    for c in range(grid_x):
        for r in range(grid_y):
            print(max(data[i][:,c]) - min(data[i][:,c]))

# Create the animation
ani = FuncAnimation(fig, update, frames=num_timesteps, blit=True, interval=200)

# Show the animation
plt.colorbar(heatmap)
plt.show()