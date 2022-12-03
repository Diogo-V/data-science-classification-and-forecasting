import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import numpy as np
from ds_charts import get_variable_types
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

    # Builds datasets with both approaches
    zscore = self.scale_zscore()
    min_max = self.scale_mim_max()

    zscore = zscore.drop(columns=['date'])
    min_max = min_max.drop(columns=['date'])

    # Applies NB and KNN to check which one is better
    zscore_nb_acc, zscore_knn_acc = self.compute_naive_bayes_result(zscore), self.compute_knn_result(zscore)
    min_max_nb_acc, min_max_knn_acc = self.compute_naive_bayes_result(min_max), self.compute_knn_result(min_max)

    print("############ Result #############")
    print(f"ZSCORE -> NB: {zscore_nb_acc} | KNN: {zscore_knn_acc}")
    print(f"MIN-MAX -> NB: {min_max_nb_acc} | KNN: {min_max_knn_acc}")
    print("#################################")

    print("GETTING BEST NEIGHBOR VALUE:")
    self.compute_best_knn_neighbor(min_max)

  def scale_mim_max(self) -> pd.DataFrame:

    variable_types = get_variable_types(self.data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = self.data[numeric_vars]
    df_sb = self.data[symbolic_vars]
    df_bool = self.data[boolean_vars]

    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = pd.DataFrame(transf.transform(df_nr), index=self.data.index, columns= numeric_vars)
    norm_data_minmax = pd.concat([tmp, df_sb,  df_bool], axis=1)

    return norm_data_minmax

  def scale_zscore(self) -> pd.DataFrame:

    variable_types = get_variable_types(self.data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = self.data[numeric_vars]
    df_sb = self.data[symbolic_vars]
    df_bool = self.data[boolean_vars]

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = pd.DataFrame(transf.transform(df_nr), index=self.data.index, columns=numeric_vars)
    norm_data_zscore = pd.concat([tmp, df_sb,  df_bool], axis=1)
    return norm_data_zscore

  def compute_naive_bayes_result(self, dataset: pd.DataFrame) -> float:
    """
    Description:
      * Computes Naive Bayes accuracy using StratifiedKFold.

    Arguments:
      * dataset(pd.DataFrame): dataset to be analyzed using a naive bayes approach

    Returns:
      * float: accuracy found
    """

    # Gets sub datasets to test
    X = dataset.drop("class", axis=1).to_numpy()
    y = dataset["class"].to_numpy()

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

  def compute_knn_result(self, dataset: pd.DataFrame) -> float:
    """
    Description:
      * Computes KNN accuracy using StratifiedKFold with a different number of neighbors.

    Arguments:
      * dataset(pd.DataFrame): dataset to be analyzed using a knn approach

    Returns:
      * float: best accuracy found
    """

    # Gets sub datasets to test
    X = dataset.drop("class", axis=1).to_numpy()
    y = dataset["class"].to_numpy()

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

  def compute_best_knn_neighbor(self, dataset: pd.DataFrame):
    """
    Description:
      * Decides the best neighbor size for KNN.

    Arguments:
      * dataset(pd.DataFrame): dataset to be analyzed using a knn approach
    """

    # Gets sub datasets to test
    X = dataset.drop("class", axis=1).to_numpy()
    y = dataset["class"].to_numpy()

    test_neighbors = [3, 5, 7, 9, 11, 13, 15]

    # We need to create a classifier for each number of neighbors
    for n in test_neighbors:

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

          # Uses training data and gets model accuracy to determine over fitting
          acc = clf.score(X_train, y_train)
          train_acc.append(acc)

      # Calculates means for train and test to determine which one is over fitting less
      train_mean = sum(train_acc) / 10
      test_mean = sum(test_acc) / 10
      error = math.sqrt(np.square(np.subtract(train_acc, test_acc)).mean())
      print("Training acc: {:.3f}".format(train_mean))
      print("Test acc: {:.3f}".format(test_mean))
      print("Diff between train and test: {:.3f}".format(train_mean - test_mean))
      print("Root mean squared error: {:.3f}".format(error))
      print("########################")

  def compute_ttest(self, dataset: pd.DataFrame):
    # Gets sub datasets to test
    X = dataset.drop("class", axis=1).to_numpy()
    y = dataset["class"].to_numpy()

    # Holds accuracy for each model to be latter used in t-test
    knn_acc = []
    mnb_acc = []

    # Creates a k fold cross validator
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.DETERMINISM_FACTOR)

    # Creates KNN classifier for 3 neighbours
    knn = KNeighborsClassifier(9, weights="uniform", p=2, metric="minkowski")

    # Creates a Multinomial Naive Bayes classifier (since the question tells us to use "multinomial assumption")
    mnb = GaussianNB()

    # For each train/test set, we use a KNN classifier
    for train_index, test_index in skf.split(X, y):
    
        # Uses indexes to fetch which values are going to be used to train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Trains knn classifier
        knn.fit(X_train, y_train.ravel())

        # Uses testing data and gets model accuracy
        acc = knn.score(X_test, y_test)

        # Appends accuracy to be latter used as input in a t-test to compare with gnb
        knn_acc.append(acc)

        # Trains gnb classifier
        mnb.fit(X_train, y_train.ravel())

        # Uses testing data and gets model accuracy
        acc = mnb.score(X_test, y_test)

        # Appends accuracy to be latter used as input in a t-test to compare with knn
        mnb_acc.append(acc)

    # Uses a t-test to compare both models and determine which one is better
    statistic, p_value = ttest_rel(knn_acc, mnb_acc, nan_policy="omit", alternative="two-sided")

    print(f"statistic: {statistic} | p_value: {p_value}")
