import numpy as np
import matplotlib.pyplot as plt

# Define the range of x and y
x = np.arange(0, 1, 0.1)
y = np.arange(0, 1, 0.1)

# Define the min and max values for the mesh grid
x_min, x_max = x.min() - 1, x.max() + 1
y_min, y_max = y.min() - 1, y.max() + 1

# Choose a step size (either use a fixed value or calculate it)
# Option 1: Fixed step size based on original x and y spacing
# step = 0.1  # Original step size of x and y arrays

# Option 2: Step size based on the range of x and y
step = (x_max - x_min) / 100  # Choose 100 intervals for the grid

# Create the mesh grid
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step)) # Creating a 

# Print the shapes to confirm
print(f"The shape of x is {x.shape}, while xx is {xx.shape}")
print(f"The shape of y is {y.shape}, while yy is {yy.shape}")

print(np.c_[xx.ravel(), yy.ravel()])
