import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import numpy as np
from scipy.stats import ttest_rel


class Scaling:

  DETERMINISM_FACTOR = 3
  NEIGHBORS = [7, 9]
  
  def __init__(self, data: pd.DataFrame) -> None:
    """
    Description:
      * Builds a scaler object that brings the dataset into a more concise space.

    Arguments:
      * data(pd.DataFrame): dataset to be treated
    """
    self.data: pd.DataFrame = data

  def analyze_scaling(self):
    self.data = self.data.drop(columns=['date'])

    X = self.data.drop("class", axis=1).to_numpy()
    y = self.data["class"].to_numpy()

    acc_knn_min_max, acc_nb_min_max = self.scale_min_max(X, y)
    acc_knn_z_score, acc_nb_z_score = self.scale_zscore(X, y)

    print(f"MIN-MAX -> Accuracy for KNN: {acc_knn_min_max} | Accuracy for NB: {acc_nb_min_max}")
    print(f"Z SCORE -> Accuracy for KNN: {acc_knn_z_score} | Accuracy for NB: {acc_nb_z_score}")

  def scale_zscore(self, X, y):
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    acc_knn = self.compute_knn_result(X, y)
    acc_nb = self.compute_naive_bayes_result(X, y)

    return acc_knn, acc_nb

  def scale_min_max(self, X, y):
    """
    Description:
      * Scales dataset using a min-max algorithm into a given range.
    """
    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    acc_knn = self.compute_knn_result(X, y)
    acc_nb = self.compute_naive_bayes_result(X, y)
    return acc_knn, acc_nb


  def compute_naive_bayes_result(self, X: np.ndarray, y: np.ndarray) -> float:
    """
    Description:
      * Computes Naive Bayes accuracy using StratifiedKFold.

    Arguments:
      * X(np.ndarray): Input dataset that is going to be used to calculate naive bayes
      * y(np.ndarray): Output for each row (used to compare results from NB and get accuracy)

    Returns:
      * float: accuracy found
    """

    # Holds training and testing accuracy to compute mean
    train_acc = []
    test_acc = []

    # Creates a Gaussian Naive Bayes classifier
    gnb = GaussianNB()

    # Creates a k fold cross validator
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.DETERMINISM_FACTOR)

    # For each train/test set, we use a KNN classifier
    for train_index, test_index in skf.split(X, y):
    
      # Uses indexes to fetch which values are going to be used to train and test
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # Trains knn classifier
      gnb.fit(X_train, y_train.ravel())

      # Uses testing data and gets model accuracy
      acc = gnb.score(X_test, y_test)
      test_acc.append(acc)
      print("Acc using test data {:.3f}".format(acc))

      # Uses training data and gets model accuracy to determine over fitting
      acc = gnb.score(X_train, y_train)
      train_acc.append(acc)
      print("Acc using training data {:.3f}".format(acc))
    
    # Calculates means for train and test to determine which one is over fitting less
    train_mean = sum(train_acc) / 10
    test_mean = sum(test_acc) / 10
    error = math.sqrt(np.square(np.subtract(train_acc, test_acc)).mean())
    print("Training acc: {:.3f}".format(train_mean))
    print("Test acc: {:.3f}".format(test_mean))
    print("Diff between train and test: {:.3f}".format(train_mean - test_mean))
    print("Root mean squared error: {:.3f}".format(error))

    return test_mean

  def compute_knn_result(self, X: np.ndarray, y: np.ndarray) -> float:
    """
    Description:
      * Computes KNN accuracy using StratifiedKFold with a different number of neighbors.

    Arguments:
      * X(np.ndarray): Input dataset that is going to be used to calculate knn
      * y(np.ndarray): Output for each row (used to compare results from KNN and get accuracy)

    Returns:
      * float: best accuracy found
    """

    best_test_acc = -1

    # We need to create a classifier for each number of neighbors
    for n in self.NEIGHBORS:

      # Holds training and testing accuracy to be latter used to determine which K is more susceptible to over fit
      train_acc = []
      test_acc = []

      print(f"Classifying n = {n}:")

      # Creates a k fold cross validator
      skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.DETERMINISM_FACTOR)

      # Creates KNN classifier for n neighbors
      clf = KNeighborsClassifier(n, weights="uniform", p=2, metric="minkowski")

      # For each train/test set, we use a KNN classifier
      for train_index, test_index in skf.split(X, y):

          # Uses indexes to fetch which values are going to be used to train and test
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = y[train_index], y[test_index]

          # Trains knn classifier
          clf.fit(X_train, y_train.ravel())

          # Uses testing data and gets model accuracy
          acc = clf.score(X_test, y_test)
          test_acc.append(acc)
          print("Acc using test data {:.3f}".format(acc))

          # Uses training data and gets model accuracy to determine over fitting
          acc = clf.score(X_train, y_train)
          train_acc.append(acc)
          print("Acc using training data {:.3f}".format(acc))

      # Calculates means for train and test to determine which one is over fitting less
      train_mean = sum(train_acc) / 10
      test_mean = sum(test_acc) / 10
      error = math.sqrt(np.square(np.subtract(train_acc, test_acc)).mean())
      print("Training acc: {:.3f}".format(train_mean))
      print("Test acc: {:.3f}".format(test_mean))
      print("Diff between train and test: {:.3f}".format(train_mean - test_mean))
      print("Root mean squared error: {:.3f}".format(error))

      if test_mean > best_test_acc:
        best_test_acc = test_mean

    return best_test_acc
