import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Parameters
sample1 = int(input("How many samples 1 do you want? "))
sample2 = int(input("How many samples 2 do you want? "))
mean = 0
std_dev = 0.6
shape1 = (sample1, 2)
shape2 = (sample2, 2)

# Generate random values for two classes from a normal distribution
matrix1 = np.random.normal(mean, std_dev, shape1) + np.array([3, 4])
matrix2 = np.random.normal(mean, std_dev, shape2) + np.array([4, 3])

# Create labels: class 1 for matrix1 and class -1 for matrix2
labels1 = np.ones(sample1)
labels2 = -1 * np.ones(sample2)

# Combine samples and labels
matrix_concat = np.concatenate((matrix1, matrix2), axis=0)
matrix_labels = np.concatenate((labels1, labels2))

def show_graphic():
    plt.scatter(matrix1[:, 0], matrix1[:, 1], color='red', label='Class 1')
    plt.scatter(matrix2[:, 0], matrix2[:, 1], color='blue', label='Class -1')
    plt.title('Random Matrix with Mean 0 and Std Dev 0.6')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.savefig('scatter_plot.png')
    plt.close()

var_show = int(input("Do you want to show the graphic? (1 for Yes, 0 for No) "))
if var_show == 1:
    show_graphic()

# Create a new random point
new_point = np.random.normal(mean, std_dev, size=(1, 2)) + np.array([3, 4])
print(f"New random point: {new_point}")

# Gaussian Kernel function
def gaussian_kernel(distance, bandwidth=1.0):
    return np.exp(- (distance ** 2) / (2 * bandwidth ** 2)) / math.sqrt(2 * math.pi)

freq1 = matrix1.shape[0]
freq2 = matrix2.shape[0]

# KNN function to classify a new point based on the k nearest neighbors
def knn_altered_classify(matrix, freq1, freq2, labels, new_point, k):
    alfa1 = 1 / freq1
    alfa2 = 1 / freq2
    distances = np.linalg.norm(matrix - new_point, axis=1)

    distances[:freq1] *= alfa1
    distances[freq1:] *= alfa2
    
    std_dev = float(input("Type here the value of standard deviation: "))
    weights = gaussian_kernel(distances, std_dev)  
    weighted_distances = distances * weights

    nearest_indices = np.argsort(weighted_distances)[:k]
    sum_labels = np.sum(labels[nearest_indices])

    return 1 if sum_labels > 0 else -1

# Generate a mesh grid to plot decision boundary
def plot_decision_boundary(matrix, freq1, freq2, labels, k, resolution=0.1):
    x_min, x_max = matrix[:, 0].min() - 1, matrix[:, 0].max() + 1
    y_min, y_max = matrix[:, 1].min() - 1, matrix[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([knn_altered_classify(matrix, freq1, freq2, labels, point, k) for point in grid_points])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(matrix1[:, 0], matrix1[:, 1], color='red', label='Class 1')
    plt.scatter(matrix2[:, 0], matrix2[:, 1], color='blue', label='Class -1')
    plt.scatter(new_point[:, 0], new_point[:, 1], color='black', s=100, label='New Point')
    plt.title('Decision Boundary and Data Points')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.savefig('decision_boundary.png')
    plt.close()

# Get value for k from the user
k = int(input("Enter the number of neighbors (k) for KNN: "))

# Display the decision boundary
plot_decision_boundary(matrix_concat, freq1, freq2, matrix_labels, k)
print("Plot saved as 'decision_boundary.png'.")

# 3D Surface Plot Function
def plot_3d_surface(matrix, freq1, freq2, labels, k):
    x_min, x_max = matrix[:, 0].min() - 1, matrix[:, 0].max() + 1
    y_min, y_max = matrix[:, 1].min() - 1, matrix[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([knn_altered_classify(matrix, freq1, freq2, labels, point, k) for point in grid_points])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(xx, yy, Z, alpha=0.5, cmap='coolwarm')
    ax.scatter(matrix1[:, 0], matrix1[:, 1], np.ones_like(matrix1[:, 0]) * 0.5, color='red', label='Class 1')
    ax.scatter(matrix2[:, 0], matrix2[:, 1], np.ones_like(matrix2[:, 0]) * 0.5, color='blue', label='Class -1')
    ax.scatter(new_point[:, 0], new_point[:, 1], 1, color='black', s=100, label='New Point')

    ax.set_title('3D Decision Surface')
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.set_zlabel('Decision Boundary')
    ax.legend()

    plt.show()

# Display the 3D surface plot
plot_3d_surface(matrix_concat, freq1, freq2, matrix_labels, k)

bool_var = True

while bool_var:
    # Classify the new point and plot the 3D surface
    returned_val = knn_altered_classify(matrix_concat, freq1, freq2, matrix_labels, new_point, k)
    plot_3d_surface(matrix_concat, freq1, freq2, matrix_labels, k)

    # Ask user if they want to see the plot again
    bool_var = bool(int(input("Type 1 to see again or 0 to exit: ")))
