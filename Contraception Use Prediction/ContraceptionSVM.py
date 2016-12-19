import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o',s=55, label='test set')
    

def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data',header=None)
    print(df.tail())
    y = df.iloc[0:580,9].values
    y = np.where(y == 1,-1,1)
    X = df.iloc[0:580,[3,7]].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)#check if split proportions are correct
    X_test, X_dev, y_test, y_dev = train_test_split(X_test,y_test,test_size=0.5, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_dev_std = sc.transform(X_dev)

    X_combined_std = np.vstack((X_train_std, X_dev_std))
    y_combined = np.hstack((y_train, y_dev))


    




    cls = svm.SVC(kernel='rbf',random_state = 0,gamma=30,C=1.0) 
    cls.fit(X_train,y_train)

    plot_decision_regions(X_combined_std, y_combined, classifier=cls)


    plt.xlabel('Number of children ever born [standardized]')
    plt.ylabel('standard of living index [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    




main()

