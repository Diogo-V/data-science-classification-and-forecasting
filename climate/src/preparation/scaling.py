import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import numpy as np
from ds_charts import plot_confusion_matrix, plot_evaluation_results
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Scaling:

  DETERMINISM_FACTOR = 3
  NEIGHBORS = [15]  # 15 is the best neighbor
  
  def __init__(self, data: pd.DataFrame) -> None:
    """
    Description:
      * Builds a scaler object that brings the dataset into a more concise space.

    Arguments:
      * data(pd.DataFrame): dataset to be treated
    """
    self.data: pd.DataFrame = data
  
  def compute_scale(self) -> pd.DataFrame:
    return self.scale_min_max()  # min max is the best one

  def explore_scaling(self):

    # Builds datasets with both approaches
    zscore = self.scale_zscore()
    min_max = self.scale_min_max()

    zscore = zscore.drop(columns=['month', 'day', 'year'])
    min_max = min_max.drop(columns=['month', 'day', 'year'])

    # Applies NB and KNN to check which one is better
    print("COMPUTING Z-SCORE...")
    zscore_nb_acc, zscore_knn_acc = self.evaluate_nb(zscore, 'zscore'), self.evaluate_knn(zscore, 'zscore')
    print("COMPUTING MIN-MAX...")
    min_max_nb_acc, min_max_knn_acc = self.evaluate_nb(min_max, 'minmax'), self.evaluate_knn(min_max, 'minmax')

    print("############ Result #############")
    print(f"ZSCORE -> NB: {zscore_nb_acc} | KNN: {zscore_knn_acc}")
    print(f"MIN-MAX -> NB: {min_max_nb_acc} | KNN: {min_max_knn_acc}")
    print("#################################")

    print("GETTING BEST NEIGHBOR VALUE:")
    # self.compute_best_knn_neighbor(zscore)

    fig, axs = plt.subplots(1, 3, figsize=(20,10),squeeze=False)
    axs[0, 0].set_title('Original data')
    self.data.boxplot(ax=axs[0, 0], rot=45, fontsize=4)
    axs[0, 1].set_title('Z-score normalization')
    zscore.boxplot(ax=axs[0, 1], rot=45, fontsize=4)
    axs[0, 2].set_title('MinMax normalization')
    min_max.boxplot(ax=axs[0, 2], rot=45, fontsize=4)
    plt.savefig(f'climate/records/preparation/scaling_boxplots.png')

  def scale_min_max(self) -> pd.DataFrame:

    numeric_vars = ['fips', 'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'lat', 'lon', 'elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND']

    symbolic_vars = ['day', 'month', 'year', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']
    
    boolean_vars = ['class']

    df_nr = self.data[numeric_vars]
    df_sb = self.data[symbolic_vars]
    df_bool = self.data[boolean_vars]

    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = pd.DataFrame(transf.transform(df_nr), index=self.data.index, columns= numeric_vars)
    norm_data_minmax = pd.concat([tmp, df_sb,  df_bool], axis=1)

    return norm_data_minmax

  def scale_zscore(self) -> pd.DataFrame:
    numeric_vars = ['fips', 'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'lat', 'lon', 'elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND']

    symbolic_vars = ['day', 'month', 'year', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']

    boolean_vars = ['class']

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

    print("COMPUTING NB...")

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

      # Uses training data and gets model accuracy to determine over fitting
      acc = gnb.score(X_train, y_train)
      train_acc.append(acc)
    
    # Calculates means for train and test to determine which one is over fitting less
    train_mean = sum(train_acc) / 10
    test_mean = sum(test_acc) / 10
    error = math.sqrt(np.square(np.subtract(train_acc, test_acc)).mean())
    print("Training acc: {:.5f}".format(train_mean))
    print("Test acc: {:.5f}".format(test_mean))
    print("Diff between train and test: {:.5f}".format(train_mean - test_mean))
    print("Root mean squared error: {:.5f}".format(error))

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

    print("COMPUTING KNN...")

    # Gets sub datasets to test
    X = dataset.drop("class", axis=1).to_numpy()
    y = dataset["class"].to_numpy()

    best_test_acc = -1
    best_neighbor = -1

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

          # Uses training data and gets model accuracy to determine over fitting
          acc = clf.score(X_train, y_train)
          train_acc.append(acc)

      # Calculates means for train and test to determine which one is over fitting less
      train_mean = sum(train_acc) / 10
      test_mean = sum(test_acc) / 10
      error = math.sqrt(np.square(np.subtract(train_acc, test_acc)).mean())
      print("Training acc: {:.5f}".format(train_mean))
      print("Test acc: {:.5f}".format(test_mean))
      print("Diff between train and test: {:.5f}".format(train_mean - test_mean))
      print("Root mean squared error: {:.5f}".format(error))

      if test_mean > best_test_acc:
        best_test_acc = test_mean
        best_neighbor = n
    print(f"Best neighbor: {best_neighbor}")

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
      print("Training acc: {:.5f}".format(train_mean))
      print("Test acc: {:.5f}".format(test_mean))
      print("Diff between train and test: {:.5f}".format(train_mean - test_mean))
      print("Root mean squared error: {:.5f}".format(error))
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


  def evaluate_knn(self, dataset: pd.DataFrame, approach: str):
    data = dataset.copy(deep=True)
    y = data.pop('class').values
    X = data.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=self.DETERMINISM_FACTOR)

    labels = pd.unique(y)
    labels.sort()

    labels_str=["1", "2"]	

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    prd_train = knn.predict(X_train)
    prd_tst = knn.predict(X_test)
    train_acc = accuracy_score(y_train, prd_train)
    test_acc = accuracy_score(y_test, prd_tst)
    error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

    plot_evaluation_results(labels, y_train, prd_train, y_test, prd_tst)
    plt.savefig(f'climate/records/preparation/scaling_{approach}_knn_results.png')

    f= open(f'climate/records/preparation/scaling_{approach}_knn_results_details.txt', 'w')
    f.write("Accuracy Train: {:.5f}\n".format(train_acc))
    f.write("Accuracy Test: {:.5f}\n".format(test_acc))
    f.write("Diff between train and test: {:.5f}\n".format(train_acc - test_acc))
    f.write("Root mean squared error: {:.5f}\n".format(error))
    f.write("########################\n")
    f.write("Train\n")
    f.write(classification_report(y_train, prd_train,target_names=labels_str))
    f.write("Test\n")
    f.write(classification_report(y_test, prd_tst,target_names=labels_str))
    
    return test_acc

  def evaluate_nb(self, dataset: pd.DataFrame, approach: str):
    data = dataset.copy(deep=True)
    y = data.pop('class').values
    X = data.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=self.DETERMINISM_FACTOR)

    labels = pd.unique(y)
    labels.sort()

    labels_str=["1", "2"]	

    nb = GaussianNB()	
    nb.fit(X_train, y_train)
    prd_train = nb.predict(X_train)
    prd_tst = nb.predict(X_test)
    train_acc = accuracy_score(y_train, prd_train)
    test_acc = accuracy_score(y_test, prd_tst)
    error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

    plot_evaluation_results(labels, y_train, prd_train, y_test, prd_tst)
    plt.savefig(f'climate/records/preparation/scaling_{approach}_nb_results.png')

    f= open(f'climate/records/preparation/scaling_{approach}_nb_results_details.txt', 'w')
    f.write("Accuracy Train: {:.5f}\n".format(train_acc))
    f.write("Accuracy Test: {:.5f}\n".format(test_acc))
    f.write("Diff between train and test: {:.5f}\n".format(train_acc - test_acc))
    f.write("Root mean squared error: {:.5f}\n".format(error))
    f.write("########################\n")
    f.write("Train\n")
    f.write(classification_report(y_train, prd_train,target_names=labels_str))
    f.write("Test\n")
    f.write(classification_report(y_test, prd_tst,target_names=labels_str))

    return test_acc