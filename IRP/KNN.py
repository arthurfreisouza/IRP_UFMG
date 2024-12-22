import numpy as np
import matplotlib.pyplot as plt

# Optional: Set the Matplotlib backend (comment out if it causes issues)
# import matplotlib
# matplotlib.use('TkAgg')  # Uncomment this line if you're using a TkAgg supported environment

# Parameters
sample = int(input("How many samples do you want? "))
mean = 0
std_dev = 0.6
shape = (sample, 2)

# Generate random values for two classes from a normal distribution
matrix1 = np.random.normal(mean, std_dev, shape) + np.array([3, 4])
matrix2 = np.random.normal(mean, std_dev, shape) + np.array([4, 3])

# Create labels: class 1 for matrix1 and class -1 for matrix2
labels1 = np.ones((sample, 1))
labels2 = -1 * np.ones((sample, 1))

# Combine samples and labels
matrix_concat = np.concatenate((matrix1, matrix2), axis=0)
matrix_labels = np.concatenate((labels1, labels2), axis=0).reshape(-1)

# Plot the initial scatter plot of the data
def show_graphic():
    plt.scatter(matrix1[:, 0], matrix1[:, 1], color='red', label='Class 1')
    plt.scatter(matrix2[:, 0], matrix2[:, 1], color='blue', label='Class -1')
    plt.title('Random Matrix with Mean 0 and Std Dev 0.6')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.savefig('scatter_plot.png')  # Save the plot as an image
    plt.close()  # Close the plot to free up memory

var_show = int(input("Do you want to show the graphic? (1 for Yes, 0 for No) "))
if var_show == 1:
    show_graphic()  # Optionally show the graphic in a supported environment

# Create a new random point
new_point = np.random.normal(mean, std_dev, size=(1, 2)) + np.array([3, 4])
print(f"New random point: {new_point}")

# KNN function to classify a new point based on the k nearest neighbors
def knn_classify(matrix, labels, new_point, k):
    distances = np.linalg.norm(matrix - new_point, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    sum_labels = np.sum(labels[nearest_indices])
    return 1 if sum_labels > 0 else -1

# Generate a mesh grid to plot decision boundary
def plot_decision_boundary(matrix, labels, k, resolution=0.1):
    # Define the range for the grid
    x_min, x_max = matrix[:, 0].min() - 1, matrix[:, 0].max() + 1
    y_min, y_max = matrix[:, 1].min() - 1, matrix[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    # Flatten the grid to get all (x, y) pairs
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict class for each point in the grid
    Z = np.array([knn_classify(matrix, labels, point, k) for point in grid_points])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(matrix1[:, 0], matrix1[:, 1], color='red', label='Class 1')
    plt.scatter(matrix2[:, 0], matrix2[:, 1], color='blue', label='Class -1')
    plt.scatter(new_point[:, 0], new_point[:, 1], color='black', s=100, label='New Point')
    plt.title('Decision Boundary and Data Points')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.savefig('decision_boundary.png')  # Save the decision boundary plot as an image
    plt.close()  # Close the plot to free up memory

# Get value for k from the user
k = int(input("Enter the number of neighbors (k) for KNN: "))

# Display the decision boundary
plot_decision_boundary(matrix_concat, matrix_labels, k)

print("Plots saved as 'scatter_plot.png' and 'decision_boundary.png'.")
