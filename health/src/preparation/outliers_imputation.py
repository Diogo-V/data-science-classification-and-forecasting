import pandas as pd
from ds_charts import get_variable_types
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import math
import numpy as np
from ds_charts import get_variable_types

OUTLIER_PARAM: int = 2 # define the number of stdev to use or the IQR scale (usually 1.5)
OPTION = 'stddev'  # or 'stdev'

class OutliersImputation:
  
  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data

  def determine_outlier_thresholds(self, summary5: pd.DataFrame, var: str):
    if 'iqr' == OPTION:
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%']  + iqr
        bottom_threshold = summary5[var]['25%']  - iqr
    else:  # OPTION == 'stdev'
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    return top_threshold, bottom_threshold

  def compute_outliers(self) -> pd.DataFrame:
    return self.compute_drop_outliers()  # This is the best one

  def explore_outliers(self) -> pd.DataFrame:
    data_drop = self.compute_drop_outliers()
    data_median = self.compute_median_outliers()
    data_truncate = self.compute_truncate_outliers()

    print("TESTING DROP OUTLIERS")
    self.compute_knn_result(data_drop)
    print("TESTING MEDIAN OUTLIERS")
    self.compute_knn_result(data_median)
    print("TESTING TRUNCATE OUTLIERS")
    self.compute_knn_result(data_truncate)

  def compute_drop_outliers(self) -> pd.DataFrame:
    numeric_vars = get_variable_types(self.data)['Numeric']
    numeric_vars.remove("readmitted")
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    print('Original data:', self.data.shape)
    summary5 = self.data.describe(include='number')
    df = self.data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = self.determine_outlier_thresholds(summary5, var)
        outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
    print('data after dropping outliers:', df.shape)
    return df

  def compute_median_outliers(self) -> pd.DataFrame:
    numeric_vars = get_variable_types(self.data)['Numeric']
    numeric_vars.remove("readmitted")
    if [] == numeric_vars:
      raise ValueError('There are no numeric variables.')

    summary5 = self.data.describe(include='number')
    df = self.data.copy(deep=True)
    for var in numeric_vars:
      top_threshold, bottom_threshold = self.determine_outlier_thresholds(summary5, var)
      median = df[var].median()
      df[var] = df[var].apply(lambda x: median if x > top_threshold or x < bottom_threshold else x)

    print('data after replacing outliers:', df.shape)
    return df

  def compute_truncate_outliers(self) -> pd.DataFrame:
    numeric_vars = get_variable_types(self.data)['Numeric']
    numeric_vars.remove("readmitted")
    if [] == numeric_vars:
      raise ValueError('There are no numeric variables.')

    summary5 = self.data.describe(include='number')
    df = self.data.copy(deep=True)
    for var in numeric_vars:
      top_threshold, bottom_threshold = self.determine_outlier_thresholds(summary5, var)
      df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)

    print('data after truncating outliers:', df.shape)
    return df

  def compute_knn_result(self, dataset: pd.DataFrame) -> tuple[float, float]:
    """
    Description:
      * Computes KNN accuracy using StratifiedKFold with a different number of neighbors.
  
    Arguments:
      * dataset(pd.DataFrame): dataset to be analyzed using a knn approach
  
    Returns:
      * float: best accuracy found
    """
  
    # Gets sub datasets to test
    X = dataset.drop("readmitted", axis=1).to_numpy()
    y = dataset["readmitted"].to_numpy()
  
    best_test_acc = -1
    best_k_value = -1
  
    # We need to create a classifier for each number of neighbors
    for n in [7, 9, 11]:
    
      # Holds training and testing accuracy to be latter used to determine which K is more susceptible to over fit
      train_acc = []
      test_acc = []
  
      print(f"Classifying n = {n}:")
  
      # Creates a k fold cross validator
      skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
  
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
  
      if test_mean > best_test_acc:
        best_test_acc = test_mean
        best_k_value = n
  
    return best_test_acc, best_k_value