import pandas as pd
from matplotlib.pyplot import figure, xticks, savefig
from ts_functions import plot_series, HEIGHT
import matplotlib.pyplot as plt
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, HEIGHT
from sklearn.base import RegressorMixin


class Differentiation:

  def __init__(self, data: pd.DataFrame) -> None:
      self.data: pd.DataFrame = data

  def compute_differentiation(self) -> pd.DataFrame:
    return self.compute_first_diff(self.data)  # Seems like it's the best one

  def explore_differentiation(self) -> None:
    no_diff = self.compute_no_diff(self.data)
    first_diff = self.compute_first_diff(self.data)
    second_diff = self.compute_second_diff(self.data)

    # Plots figures for each differentiation measure
    self.plot_figure(no_diff, 'no_diff', 'no differentiation')
    self.plot_figure(first_diff, 'first_diff', 'first differentiation')
    self.plot_figure(second_diff, 'second_diff', 'second differentiation')

    # Evaluates results with MA and RM
    # print("EVALUATING MODELS WITH SMA...")
    # self.simple_average(no_diff, 'no_diff')
    # self.simple_average(first_diff, 'first_diff')
    # self.simple_average(second_diff, 'second_diff')

    print("EVALUATING MODELS WITH PERSISTANCE...")
    self.persistance(no_diff, 'no_diff', 'no differentiation')
    self.persistance(first_diff, 'first_diff', 'first differentiation')
    self.persistance(second_diff, 'second_diff', 'second differentiation')

    # print("EVALUATING MODELS WITH RM...")
    # self.rolling_mean(no_diff, 'no_diff')
    # self.rolling_mean(first_diff, 'first_diff')
    # self.rolling_mean(second_diff, 'second_diff')

  def plot_figure(self, data: pd.DataFrame, approach: str, title: str) -> None:
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(data, title=title, x_label='Date', y_label='QV2M')
    xticks(rotation = 45)
    savefig(f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_plot.png')

  def compute_no_diff(self, data: pd.DataFrame) -> pd.DataFrame:
    return data  # Just for compliance

  def compute_first_diff(self, data: pd.DataFrame) -> pd.DataFrame:
    diffed = data.diff(periods=1)
    res = diffed.dropna(inplace=False)
    return res 

  def compute_second_diff(self, data: pd.DataFrame) -> pd.DataFrame:
    diffed = data.diff(periods=2)
    res = diffed.dropna(inplace=False)
    return res

  def split_dataframe(self, data: pd.DataFrame, trn_pct=0.70):
      trn_size = int(len(data) * trn_pct)
      df_cp = data.copy()
      train: pd.DataFrame = df_cp.iloc[:trn_size, :]
      test: pd.DataFrame = df_cp.iloc[trn_size:]
      return train, test

  def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, file_path: str, tittle: str, x_label: str = 'time', y_label:str =''):
      _, ax = plt.subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.set_title(tittle)
      ax.plot(trn.index, trn, label='train', color='b')
      ax.plot(trn.index, prd_trn, '--y', label='train prediction')
      ax.plot(tst.index, tst, label='test', color='g')
      ax.plot(tst.index, prd_tst, '--r', label='test prediction')
      ax.legend(prop={'size': 5})
      savefig(file_path)

  def persistance(self, data: pd.DataFrame, approach: str, tittle: str) -> None:
  
      train, test = self.split_dataframe(data, trn_pct=0.75)
      eval_results = {}

      fr_mod = PersistenceRegressor()
      fr_mod.fit(train)
      prd_trn = fr_mod.predict(train)
      prd_tst = fr_mod.predict(test)

      measure = "R2"
      eval_results['Persistance'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
      print(eval_results)
      
      plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, tittle, f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_{measure}_persistance_eval.png')
      self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_{measure}_persistance_plots.png', tittle = tittle, x_label="date", y_label="QV2M")

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

      plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_{measure}_simple_avg_eval')

      self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_{measure}_simple_avg_plots.png', f'{approach} {measure}', x_label="date", y_label="QV2M")
      plt.savefig(f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_{measure}_simple_avg_plots.png')


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
    
    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'climate-forecasting/records/transformation/differentiation/{approach}_{measure}_rollingMean_eval.png')
    plt.savefig(f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_{measure}_rollingMean_eval.png')
    self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_{measure}_rollingMean_plots.png', x_label="date", y_label="QV2M")
    plt.savefig(f'climate-forecasting/records/transformation/differentiation/differentiation_{approach}_{measure}_rollingMean_plots.png')


class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: pd.DataFrame):
        self.mean = X.mean()

    def predict(self, X: pd.DataFrame):
        prd = len(X) * [self.mean]
        return prd

class PersistenceRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: pd.DataFrame):
        self.last = X.iloc[-1,0]
        print(self.last)

    def predict(self, X: pd.DataFrame):
        prd = X.shift().values
        prd[0] = self.last
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
