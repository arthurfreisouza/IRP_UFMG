import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ARTHUR_KNN import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.ion()  # Enable interactive mode

iris = datasets.load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = np.random.randint(0, 301))


def plot_grafic(X, y):
    plt.figure()
    plt.scatter(X[ :, 2], X[ :, 3], c = y, cmap = cmap, edgecolors = 'k', s = 20)
    # Save the figure
    plt.savefig('my_figure.png')  # Save as PNG

plot_grafic(X_train, y_train)

clf = KNN(k = 7)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)