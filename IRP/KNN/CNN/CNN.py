# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import CondensedNearestNeighbour
from collections import Counter

# Generating a synthetic dataset as an example
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Original class distribution
print("Original class distribution:", Counter(y))

# Applying CondensedNearestNeighbour
cnn = CondensedNearestNeighbour()
X_resampled, y_resampled = cnn.fit_resample(X_train, y_train)

# Condensed class distribution
print("Condensed class distribution:", Counter(y_resampled))

# Training a classifier with the condensed dataset
knn_original = KNeighborsClassifier()
knn_original.fit(X_train, y_train)

# Training a classifier with the condensed dataset
knn_cnn = KNeighborsClassifier()
knn_cnn.fit(X_resampled, y_resampled)

# Visualizing the results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the original dataset
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', edgecolors='k')
axes[0].set_title("Original Dataset")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

# Plotting the condensed dataset after CNN
axes[1].scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, cmap='coolwarm', marker='o', edgecolors='k')
axes[1].set_title("Condensed Dataset (after CNN)")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
