import pandas as pd
import seaborn as sns
import numpy as np

#import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# Worked with Lucas and Marissa

def loadData(datafile):
        with open(datafile, 'r', encoding='latin1') as csvfile:
            data = pd.read_csv(csvfile)

        # Inspect the data
        print(data.columns.values)

        return data

def runKNN(dataset, prediction, ignore, neighbors):
    # Set up our generateDataset
    X = dataset.drop(columns = [prediction, ignore])
    Y = dataset[prediction].values

    # Split the data into a training and testing Set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4,
    random_state=1, stratify=Y)

    # Run k-NN algorithm
    knn = KNeighborsClassifier(n_neighbors=neighbors)

    # Train the model
    knn.fit(X_train, Y_train)

    predict = knn.predict(X_test)

    #3
    # Test the model
    score = knn.score(X_test, Y_test)
    f1 = f1_score(Y_test, predict, average='macro')
    print('Predicts ' + prediction + ' with ' +str(score) + ' accuracy ')
    print('Chance is: ' +str(1.0/len(dataset.groupby(prediction))))
    print('F1 Score is: ' + str(f1))

    return knn

# The F1 score and accuracy score were both slightly less than 50%.  This means
# that using k-NN to for this particular classification would not be very
# effective because if it was the F1 value would be close to 1.

#1
def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns = [prediction, ignore])

    #2
    # Determine the five closest neighbors
    neighbors = model.kneighbors(X, n_neighbors=5, return_distance=False)

    # Print out the neighbors data
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])

#4
def kNNCrossfold(dataset, prediction, ignore, neighbors):
    fold = 0
    k_accuracies = []
    k_fold = KFold(n_splits=neighbors)

    #from previous problems: Setting up X and Y
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values

    for train,test in k_fold.split(X): #make a for loop for each kfold plit
        fold += 1 #adds to the counter we've already made
        knn = KNeighborsClassifier(n_neighbors= neighbors) #uses the k_neighbors classifier for each input
        knn = KNeighborsClassifier(n_neighbors = neighbors) #uses the kneighbors classifier for each of the k's (the 5,7,10).
        knn.fit(X[train[0]:train[-1]], Y[train[0]:train[-1]]) #fit the data in order to train the classifier on each of the folds. It will remove the last one for testing.

        predict = knn.predict(X[test[0]:test[-1]]) #knn.predict will predict the class labels for the provided the data. THis will predict the class labels for all the data in X from "column" 0 all the way to "column" -1, aka all but the last one because you're testing on the last one.
        accuracy = accuracy_score(predict, Y[test[0]:test[-1]])
        k_accuracies.append(accuracy)
        print("Fold " + str(fold) + ":" + str(accuracy))

    return np.mean(k_accuracies)

#5
def determineK(dataset, prediction, ignore, k_vals):
    optimal_k = 0
    highest_accuracy = 0

    for k in k_vals:
        initial_k = kNNCrossfold(dataset, prediction, ignore, k)
        if initial_k > highest_accuracy:
            optimal_k = k
            highest_accuracy = initial_k

    print("Best k and accuracy = " + str(optimal_k) + ", " + str(highest_accuracy))

#6
def runKMeans(dataset, ignore, neighbors):
    # Set up our dataset
    X = dataset.drop(columns=ignore)

    # Run k-Means algorithm
    kmeans = KMeans(n_clusters=neighbors)

    # Train the model
    kmeans.fit(X)

    # Add the predictions to the dataframe
    dataset['cluster'] = pd.Series(kmeans.predict(X), index =dataset.index)

    # Print a scatterplot matrix
    scatterMatrix = sns.pairplot(dataset.drop(columns=ignore), hue='cluster',
    palette='Set2')

    scatterMatrix.savefig('kmeansClusters.png')

    return kmeans

#7
def findClusterK(dataset, ignore):
    average_distances = {}
    X = dataset.drop(columns =ignore)

    for n in np.arange(4,12):
        model = runKMeans(dataset, ignore, n)
        average_distances[n] = np.mean([np.min(x) for x in model.transform(X)])

    print("Best k according to distance " + str(min(average_distances,
    key=average_distances.get)))



# Test your code
nbaData = loadData('nba_2013_clean.csv')

knnModel = runKNN(nbaData, 'pos', 'player',7)

for k in [5,7,10]:
    print("Folds: " + str(k))
    kNNCrossfold(nbaData, "pos", "player", k)

determineK(nbaData, "pos", "player", [5,7,10])

KMeansmodel = runKMeans(nbaData, ['pos', 'player'], 5)

findClusterK(nbaData, ['pos', 'player'])
