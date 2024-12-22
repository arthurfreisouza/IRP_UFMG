# Step 1: Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

# Step 2: Generate synthetic data or use your own dataset
# Let's generate a sample dataset using make_blobs
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Step 3: Apply K-Means clustering
n_clusters = 4  # Define the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Step 4: Calculate the silhouette score
# Silhouette score for the entire dataset
silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

# Calculate the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)

# Step 5: Visualize the silhouette scores (optional)
# Create a silhouette plot
fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for each cluster
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    # Calculate the position for the plot
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    # Fill the silhouette plot with color
    color = plt.cm.nipy_spectral(float(i) / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10  # 10 for space between clusters

# Set labels and title
ax.set_title(f"Silhouette Plot for {n_clusters} Clusters")
ax.set_xlabel("Silhouette Coefficient Values")
ax.set_ylabel("Cluster Label")
ax.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.savefig("a.jpeg")
