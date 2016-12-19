import math
import operator

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib.colors import ListedColormap


def euclideanDistance(x1, x2, length):
    distance = 0
    for x in range(length):
        distance+=pow((x1[x] - x2[x]),2)
    return math.sqrt(distance)

def getNeighbors(training, test, k):
    distances = []
    length = len(test) - 1
    for x in range(len(training)):
        dist = euclideanDistance(test,training[x], length)
        distances.append((training[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResult(neighbors):
    votes= {}
    for x in range(len(neighbors)):
        result = neighbors[x][-1]
        if result in votes:
            votes[result] +=1
        else:
            votes[result] = 1
    sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse = True)
    return sortedVotes[0][0]
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if (abs(testSet[x][-1] - predictions[x]) < 0.001):
            correct += 1
    return (correct/float(len(testSet)))*100.0

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
    X_test_std = sc.transform(X_test)

    print('Train set: '+repr(len(X_train_std)))
    print('Test set: '+repr(len(X_test_std)))

    predictions = []
    k = 17
    for x in range(len(X_test_std)):
        neighbors = getNeighbors(X_train_std, X_test_std[x],k)
        result = getResult(neighbors)
        predictions.append(result)
        print('> predicted='+repr(result)+', actual='+repr(X_test_std[x][-1]))
    accuracy = getAccuracy(X_test_std, predictions)
    print('Accuracy: '+repr(accuracy)+'%')
    
    


main()
