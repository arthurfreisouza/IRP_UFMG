import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
def carregar_df():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    return X, y

def plot_graphic(Z : np.ndarray = None, X : np.ndarray = None, linear : bool = None):

    plt.subplot(1, 1, 1)
    plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    if linear == True:
        plt.savefig('SVM_linear.jpg')
    else: 
        plt.savefig('SVM_nolinear.jpg')

if __name__ == "__main__":
    X, y = carregar_df()
    svc = svm.SVC(kernel = 'linear', C = 1, gamma= 0).fit(X, y)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = (x_max / x_min) / 100

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # Creating a 2D arra


    # Building a linear SVM :

    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]) # Creating a 2D array of the same points, stacked by columns. It is a grid of values.
    Z = Z.reshape(xx.shape)
    plot_graphic(Z = Z, X = X, linear = True)

    
