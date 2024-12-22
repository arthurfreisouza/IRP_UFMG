import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__ (self, k = 7):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X): # Just will calculate the predictions.
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the closest K value.
        indices = np.argsort(distances)[ : self.k]
        k_nearest_labels = [self.y_train[i] for i in indices]

        # Getting the most common.
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]  # Return the most common label
