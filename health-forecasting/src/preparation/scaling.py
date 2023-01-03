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
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, HEIGHT
from sklearn.base import RegressorMixin

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
    self.numeric_vars = ['Insulin']
    self.target_vars = ['Glucose']
  
  def compute_scale(self) -> pd.DataFrame:
    return self.scale_min_max()  # min max is the best one

  def explore_scaling(self):

    # Builds datasets with both approaches
    zscore = self.scale_zscore()
    min_max = self.scale_min_max()

    # Applies MA and RM to minimize the MSE, MAE and R2 score
    print("COMPUTING Z-SCORE WITH SMA...")
    self.simple_average(zscore, 'zscore')
    
    print("COMPUTING Z-SCORE WITH RM...")
    self.rolling_mean(zscore, 'zscore')

    print("COMPUTING MIN-MAX WITH SMA...")
    self.simple_average(min_max, 'min_max')

    print("COMPUTING Z-SCORE WITH RM...")
    self.rolling_mean(min_max, 'min_max')

  def scale_min_max(self) -> pd.DataFrame:

    df_nr = self.data[self.numeric_vars]
    df_target = self.data[self.target_vars]

    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = pd.DataFrame(transf.transform(df_nr), index=self.data.index, columns=self.numeric_vars)
    norm_data_minmax = pd.concat([tmp, df_target], axis=1)

    return norm_data_minmax

  def scale_zscore(self) -> pd.DataFrame:

    df_nr = self.data[self.numeric_vars]
    df_target = self.data[self.target_vars]

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = pd.DataFrame(transf.transform(df_nr), index=self.data.index, columns=self.numeric_vars)
    norm_data_zscore = pd.concat([tmp, df_target], axis=1)
    return norm_data_zscore

  def split_dataframe(self, data: pd.DataFrame, trn_pct=0.70):
      trn_size = int(len(data) * trn_pct)
      df_cp = data.copy()
      train: pd.DataFrame = df_cp.iloc[:trn_size, :]
      test: pd.DataFrame = df_cp.iloc[trn_size:]
      return train, test

  def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, figname: str, x_label: str = 'time', y_label:str =''):
      _, ax = plt.subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.set_title(figname)
      ax.plot(trn.index, trn, label='train', color='b')
      ax.plot(trn.index, prd_trn, '--y', label='train prediction')
      ax.plot(tst.index, tst, label='test', color='g')
      ax.plot(tst.index, prd_tst, '--r', label='test prediction')
      ax.legend(prop={'size': 5})

  def simple_average(self, data: pd.DataFrame, approach: str):

      train, test = self.split_dataframe(data, trn_pct=0.75)
      eval_results = {}

      fr_mod = SimpleAvgRegressor()
      fr_mod.fit(train)
      prd_trn = fr_mod.predict(train)
      prd_tst = fr_mod.predict(test)

      measure = "R2"

      eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
      print(eval_results)

      plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'climate-forecasting/records/preparation/{approach}_{measure}_simple_avg_eval')
      plt.savefig(f'climate-forecasting/records/preparation/{approach}_{measure}_simple_avg_eval.png')
      self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'climate-forecasting/records/preparation/{approach}_{measure}_simple_avg_plots', x_label="date", y_label="QV2M")
      plt.savefig(f'climate-forecasting/records/preparation/{approach}_{measure}_simple_avg_plots.png')

  def rolling_mean(self, data: pd.DataFrame, approach: str) -> None:
    
    train, test = self.split_dataframe(data, trn_pct=0.75)
    eval_results = {}

    fr_mod = RollingMeanRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train)
    prd_tst = fr_mod.predict(test)

    measure = "R2"
    eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
    print(eval_results)
    
    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'climate-forecasting/records/preparation/{approach}_{measure}_rollingMean_eval.png')
    plt.savefig(f'climate-forecasting/records/preparation/{approach}_{measure}_rollingMean_eval.png')
    self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'climate-forecasting/records/preparation/{approach}_{measure}_rollingMean_plots.png', x_label="date", y_label="QV2M")
    plt.savefig(f'climate-forecasting/records/preparation/{approach}_{measure}_rollingMean_plots.png')


class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: pd.DataFrame):
        self.mean = X.mean()

    def predict(self, X: pd.DataFrame):
        prd = len(X) * [self.mean]
        return prd


class RollingMeanRegressor (RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win

    def fit(self, X: pd.DataFrame):
        None

    def predict(self, X: pd.DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd
